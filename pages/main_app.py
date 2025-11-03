
"""
Main app.py — Milestone 4 (Refactored)

FIX: Moved the "Choose Column" panel logic from the top of the tab
     to *inside* the dataset loop. This makes the UI appear
     contextually, right below the "Load" button, as requested.
"""

# --- 1. IMPORTS & PAGE GUARD ---
import streamlit as st
import pandas as pd
import networkx as nx
import hashlib
import datetime
import sqlite3
import io
import os
import zipfile
import time
from typing import List

# Check if user is logged in. If not, redirect to login page.
if "username" not in st.session_state:
    st.error("You must be logged in to view this page.")
    st.page_link("app.py", label="Go to login", icon="🔒")
    st.stop()

# --- 2. PAGE CONFIG ---
st.set_page_config(
    page_title="Semantic KG Explorer — Milestone 4", 
    layout="wide"
)

# CSS hack to hide the default page navigation
st.markdown("""
    <style>
        [data-testid="stSidebarNav"] {display: none;}
    </style>
    """, unsafe_allow_html=True)


# --- 3. ALL OTHER IMPORTS ---
from config import *
from auth_db import (
    init_db, update_profile, add_feedback,
    fetch_pipeline_runs, fetch_feedback,
    add_dataset, get_datasets, get_dataset_filepath, delete_dataset
)
from kg_utils import (
    load_spacy_model, load_embed_model,
    preprocess_text,
    extract_entities_and_relations, create_neo4j_driver,
    neo4j_count_nodes_and_rels, neo4j_merge_rename, neo4j_add_relation,
    build_networkx_graph, annotate_nodes_with_entities, embed_nodes,
    semantic_query_topk, generate_union_subgraph, create_pyvis_html,
    export_graph_to_json, merge_nodes
)
from deploy_utils import write_deployment_files

# --- 4. MAIN APP LOGIC ---

# Get user info from session state
username = st.session_state["username"]
display_name = st.session_state.get("display_name", username)
is_admin = st.session_state.get("is_admin", False)

# --- Top Menu & Logout ---
st.title("Semantic Knowledge Graph Explorer — Milestone 4")
cols = st.columns([1,6,1])
with cols[0]:
    if st.button("🔒 Logout"):
        keys_to_clear = [
            'token', 'username', 'kg_graph', 'kg_nodes', 
            'kg_embeddings', 'kg_entities', 'search_results', 
            'subgraph', 'last_query', 'display_name', 'is_admin', 'theme',
            'raw_text', 'cleaned_text',
            'dataset_to_load', 'df_preview'
        ]
        for key in keys_to_clear:
            st.session_state.pop(key, None)
        st.switch_page("app.py")
with cols[1]:
    st.caption(f"Signed in as **{display_name}** ({username})")

# --- Sidebar Menu ---
menu_options = ["Graph Builder/Editor", "Feedback", "Settings"]
if is_admin:
    menu_options.insert(0, "Admin Console")

menu = st.sidebar.selectbox("Menu", menu_options)

# --- Sidebar Controls ---
st.sidebar.header("Connection & Global")
neo4j_password = st.sidebar.text_input("Neo4j Aura password (optional)", type="password")
push_to_neo4j = st.sidebar.checkbox("Push changes to Neo4j when editing", value=False)

driver = create_neo4j_driver(neo4j_password) if neo4j_password else None

st.sidebar.header("Explorer Settings")
top_k = st.sidebar.slider("Top-k", 1, 10, 3)
hop_radius = st.sidebar.slider("Subgraph hops", 0, 3, 1)
show_rel_labels = st.sidebar.checkbox("Show relation labels", True)
filter_types = st.sidebar.multiselect("Filter by entity type (optional)", ["PERSON","ORG","GPE"], default=[])

# --- Initialize Session State (for graph data) ---
state_keys = {
    'kg_graph': None, 'kg_nodes': [], 'kg_embeddings': None,
    'kg_entities': [], 'search_results': [], 'subgraph': None,
    'last_query': "",
    'raw_text': "", 
    'cleaned_text': "",
    'dataset_to_load': None, 
    'df_preview': None       
}
for key, default_val in state_keys.items():
    if key not in st.session_state:
        st.session_state[key] = default_val

# ---------------- ADMIN CONSOLE ----------------
if menu == "Admin Console":
    # ... (This section is unchanged) ...
    st.header("Admin Console")
    runs_df = fetch_pipeline_runs(500)
    fb_df = fetch_feedback(500)

    neo_stats = neo4j_count_nodes_and_rels(driver) if driver else None
    nodes_count_live = 0
    rels_count_live = 0
    if neo_stats:
         nodes_count_live = neo_stats.get('nodes', 0)
         rels_count_live = neo_stats.get('rels', 0)
    elif st.session_state['kg_graph'] is not None:
        nodes_count_live = st.session_state['kg_graph'].number_of_nodes()
        rels_count_live = st.session_state['kg_graph'].number_of_edges()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Quick metrics")
        st.metric("Entities (live)", nodes_count_live)
        st.metric("Relations (live)", rels_count_live)
        st.metric("Stored feedback", len(fb_df))
        st.markdown("---")
        st.subheader("Recent feedback")
        if not fb_df.empty:
            fb_small = fb_df.head(10).copy()
            fb_small['time'] = fb_small['timestamp'].apply(lambda x: datetime.datetime.fromtimestamp(int(x)).strftime('%Y-%m-%d %H:%M'))
            st.dataframe(fb_small[['time','username','rating','comment']])
        else:
            st.info("No feedback yet.")
    with col2:
        st.subheader("Pipeline runs & trends")
        if not runs_df.empty:
            df_vis = runs_df.copy()
            df_vis['time'] = pd.to_datetime(df_vis['timestamp'], unit='s')
            st.line_chart(df_vis.set_index('time')[['triples_count','processed_items']].fillna(0))
            st.markdown("---")
            st.write("Latest runs")
            st.dataframe(df_vis.sort_values('time', ascending=False).head(10))
        else:
            st.info("No pipeline runs logged yet.")
    st.markdown("---")
    st.subheader("Admin: Feedback export & quick actions")
    if not fb_df.empty:
        if st.button("Export feedback CSV"):
            csv_buf = fb_df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download feedback CSV", csv_buf, file_name="feedback.csv")
    else:
        st.info("No feedback to export")

# --- GRAPH BUILDER/EDITOR ---
if menu == "Graph Builder/Editor":
    
    tab_builder, tab_explorer = st.tabs(["🏗️ Graph Builder & Processor", "🔍 Semantic Search & Explore"])

    # --- BUILDER TAB (New 3-Step Workflow) ---
    with tab_builder:
        
        # --- FIX: This block has been REMOVED from the top ---
        # (The logic is now *inside* the `for` loop below)
        
        st.header("Graph Builder & Processor")
        
        # --- STEP 1: Manage Datasets ---
        st.subheader("Step 1: Manage & Load Datasets")
        
        with st.expander("Upload a new dataset (CSV)"):
            uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
            if uploaded_file:
                dataset_name = st.text_input("Enter a name for this dataset:")
                if st.button("Save Dataset"):
                    if dataset_name:
                        safe_filename = f"{username}_{int(time.time())}_{dataset_name.replace(' ', '_')}.csv"
                        file_path = os.path.join("uploads", safe_filename)
                        
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        add_dataset(username, dataset_name, file_path)
                        st.success(f"Dataset '{dataset_name}' saved!")
                        st.rerun()
                    else:
                        st.error("Please enter a dataset name.")

        st.markdown("---")
        st.write("Your saved datasets:")
        
        datasets = get_datasets(username)
        if datasets.empty:
            st.info("You have no saved datasets. Upload one above or type text below.")
        else:
            is_loading = st.session_state.dataset_to_load is not None
            
            for index, row in datasets.iterrows():
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.write(f"**{row['dataset_name']}** (Uploaded: {datetime.datetime.fromtimestamp(row['uploaded_at']).strftime('%Y-%m-%d')})")
                with col2:
                    if st.button("Load", key=f"load_{row['id']}", disabled=is_loading):
                        st.session_state.dataset_to_load = row['id']
                        st.session_state.df_preview = None 
                        st.rerun()
                with col3:
                    if st.button("Delete", key=f"del_{row['id']}", disabled=is_loading):
                        try:
                            file_path_to_delete = get_dataset_filepath(row['id'], username)
                            delete_dataset(row['id'], username)
                            if file_path_to_delete and os.path.exists(file_path_to_delete):
                                os.remove(file_path_to_delete)
                            st.success(f"Deleted '{row['dataset_name']}'.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to delete: {e}")
                
                # --- FIX: The "Choose Column" panel now appears contextually ---
                # --- right here, under the selected dataset. ---
                if st.session_state.dataset_to_load == row['id']:
                    with st.expander("Load Dataset: Choose your column", expanded=True):
                        if st.session_state.df_preview is None:
                            file_path = get_dataset_filepath(row['id'], username)
                            if file_path and os.path.exists(file_path):
                                st.session_state.df_preview = pd.read_csv(file_path)
                            else:
                                st.error("Error: Could not find dataset file.")
                                st.session_state.dataset_to_load = None
                                st.session_state.df_preview = None
                        
                        if st.session_state.df_preview is not None:
                            df = st.session_state.df_preview
                            st.write("Showing a preview of your file. Which column contains the text you want to process?")
                            st.dataframe(df.head())
                            
                            column_options = df.columns.tolist()
                            column = st.selectbox("Select text column:", column_options, key=f"select_{row['id']}")
                            
                            col_load, col_cancel = st.columns(2)
                            with col_load:
                                if st.button("Confirm and Load Column", key=f"confirm_{row['id']}"):
                                    st.session_state.raw_text = "\n\n".join(df[column].dropna().astype(str))
                                    st.session_state.cleaned_text = ""
                                    st.session_state.dataset_to_load = None
                                    st.session_state.df_preview = None
                                    st.rerun()
                            with col_cancel:
                                if st.button("Cancel", key=f"cancel_{row['id']}"):
                                    st.session_state.dataset_to_load = None
                                    st.session_state.df_preview = None
                                    st.rerun()

        # --- STEP 2: Preprocess Text ---
        st.markdown("---")
        st.subheader("Step 2: Preprocess Text")
        st.write("Type text directly, or load a dataset from above.")
        
        st.text_area("Raw Text (Input)", key="raw_text", height=200)
        
        if st.button("Run Preprocessing (lowercase, remove stop words)"):
            if st.session_state.raw_text:
                with st.spinner("Preprocessing..."):
                    st.session_state.cleaned_text = preprocess_text(st.session_state.raw_text)
                st.success("Preprocessing complete!")
            else:
                st.warning("Raw Text box is empty.")
                
        st.text_area("Cleaned Text (Output)", key="cleaned_text", height=200)

        # --- STEP 3: Build Knowledge Graph ---
        st.markdown("---")
        st.subheader("Step 3: Build Knowledge Graph")
        
        build_source = st.radio("Select build source:", ["Raw Text", "Cleaned Text"], horizontal=True)
        store_to_neo4j = st.checkbox("Push triples to Neo4j (on build)", False)

        if st.button("Run extraction & build KG"):
            # ... (This build logic is unchanged) ...
            text_to_process = ""
            if build_source == "Raw Text":
                text_to_process = st.session_state.raw_text
            else:
                text_to_process = st.session_state.cleaned_text
            
            if not text_to_process.strip():
                st.warning("No text to process. Please load a dataset or type text above.")
            else:
                text_inputs = [p.strip() for p in text_to_process.split("\n\n") if p.strip()]
                inputs_count = len(text_inputs)
                
                with st.spinner("Extracting entities and relations..."):
                    all_triples = []
                    all_entities = []
                    prog = st.progress(0)
                    for i, txt in enumerate(text_inputs):
                        ents, triples = extract_entities_and_relations(txt)
                        all_triples.extend(triples)
                        all_entities.extend(ents)
                        prog.progress((i + 1) / inputs_count)
                    
                    final_triples = list(dict.fromkeys(all_triples))
                    G = build_networkx_graph(final_triples)
                    annotate_nodes_with_entities(G, all_entities)

                    st.session_state['kg_graph'] = G
                    st.session_state['kg_nodes'] = sorted(list(G.nodes()))
                    st.session_state['kg_entities'] = all_entities
                    st.session_state['kg_embeddings'] = embed_nodes(st.session_state['kg_nodes'])
                    
                    st.success("✅ Knowledge graph built successfully!")
                    
                    try:
                        nodes_n = G.number_of_nodes()
                        edges_n = G.number_of_edges()
                        triples_count = len(final_triples)
                        acc = (len(all_entities) / max(1, inputs_count)) * 10
                        if acc > 100: acc = 100.0
                        
                        con = sqlite3.connect(DB_FILE)
                        cu = con.cursor()
                        cu.execute("""
                            INSERT INTO pipeline_runs
                            (username, timestamp, inputs_count, triples_count, nodes_count, edges_count, status, details, processed_items, extraction_accuracy)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (username, int(datetime.datetime.now().timestamp()), inputs_count, triples_count, nodes_n, edges_n, 'SUCCESS', '', triples_count, acc))
                        con.commit()
                        con.close()
                    except Exception as e:
                        st.warning(f"Failed to log pipeline run: {e}")

                    if store_to_neo4j and driver:
                        st.write("Pushing to Neo4j...")
                        try:
                            for s, r, o in final_triples:
                                neo4j_add_relation(driver, s, r, o)
                            st.success("Pushed triples to Neo4j")
                        except Exception as e:
                            st.warning(f"Pushing to Neo4j failed: {e}")
            
            st.markdown("### Extracted Named Entities")
            if 'kg_entities' in st.session_state and st.session_state['kg_entities']:
                ents_df = pd.DataFrame(st.session_state['kg_entities'], columns=["text","label"]).drop_duplicates().reset_index(drop=True)
                st.dataframe(ents_df)
            else:
                st.info("No named entities were extracted.")

    # --- EXPLORER TAB (Unchanged) ---
    with tab_explorer:
        st.header("Explore Graph — Semantic Search & Subgraph")
        if st.session_state['kg_graph'] is None:
            st.info("Build or load a graph first in the 'Graph Builder' tab.")
        else:
            # ... (All explorer tab logic is unchanged) ...
            G_full = st.session_state['kg_graph']
            nodes_all = st.session_state['kg_nodes']
            node_embs = st.session_state['kg_embeddings']

            if filter_types:
                nodes_filtered = [n for n in nodes_all if G_full.nodes[n].get('etype') in filter_types]
                if not nodes_filtered:
                    st.warning("No nodes match filter")
                    nodes_filtered = nodes_all
            else:
                nodes_filtered = nodes_all

            col1, col2, col3 = st.columns([4, 1, 1])
            with col1:
                query_text = st.text_input("Query for semantic search", value=st.session_state['last_query'])
            with col2:
                run_search = st.button("Search")
            with col3:
                if st.button("Clear Results"):
                    st.session_state['search_results'] = []
                    st.session_state['subgraph'] = None
                    st.session_state['last_query'] = ""
                    st.rerun()

            st.write(f"Nodes in graph: {G_full.number_of_nodes()}, Edges: {G_full.number_of_edges()}")

            if 'show_full_kg' not in st.session_state:
                st.session_state['show_full_kg'] = False
            
            show_btn_col1, show_btn_col2 = st.columns([1,3])
            with show_btn_col1:
                if st.button("Show Full Knowledge Graph"):
                    st.session_state['show_full_kg'] = True
            with show_btn_col2:
                if st.button("Hide Knowledge Graph"):
                    st.session_state['show_full_kg'] = False

            if st.session_state['show_full_kg']:
                try:
                    html_path = create_pyvis_html(G_full)
                    st.components.v1.html(open(html_path, 'r', encoding='utf-8').read(), height=700, scrolling=True)
                except Exception as e:
                    st.error(f"Failed to render full graph: {e}")

            if run_search and query_text.strip():
                if nodes_filtered != nodes_all and node_embs is not None:
                    search_embs = embed_nodes(nodes_filtered)
                    search_nodes = nodes_filtered
                else:
                    search_embs = node_embs
                    search_nodes = nodes_all

                if search_embs is not None:
                    results = semantic_query_topk(query_text, search_nodes, search_embs, k=top_k)
                    if not results:
                        st.info("No matches")
                        st.session_state['search_results'] = []
                        st.session_state['subgraph'] = None
                    else:
                        top_nodes = [r["node"] for r in results]
                        union_sub = generate_union_subgraph(G_full, top_nodes, hops=hop_radius)
                        if nodes_filtered != nodes_all:
                            keep = set(nodes_filtered)
                            union_sub = union_sub.subgraph([n for n in union_sub.nodes() if n in keep]).copy()
                        
                        st.session_state['search_results'] = results
                        st.session_state['subgraph'] = union_sub
                        st.session_state['last_query'] = query_text
                else:
                    st.warning("Node embeddings are not available. Please build the graph first.")

            if st.session_state['search_results']:
                st.subheader("Top matches")
                st.dataframe(pd.DataFrame(st.session_state['search_results']))
                
                union_sub = st.session_state['subgraph']
                top_nodes = [r["node"] for r in st.session_state['search_results']]
                
                if union_sub is None or union_sub.number_of_nodes() == 0:
                    st.warning("Subgraph empty after filtering")
                else:
                    html_path = create_pyvis_html(union_sub, highlight=top_nodes, show_relation_labels=show_rel_labels)
                    st.components.v1.html(open(html_path,'r',encoding='utf-8').read(), height=700, scrolling=True)
                    st.download_button("⬇️ Download subgraph JSON", export_graph_to_json(union_sub), file_name="subgraph.json", mime="application/json")

                    st.markdown("#### Manual correction & merge")
                    q_hash = hashlib.md5(st.session_state['last_query'].encode()).hexdigest()
                    edit_node = st.selectbox("Select node to edit:", sorted(list(union_sub.nodes())), key=f"edit_node_{q_hash}")
                    
                    if edit_node:
                        new_name = st.text_input("Rename node to (leave blank to keep)", value=edit_node, key=f"new_name_{edit_node}")
                        new_type = st.text_input("Set entity type (etype)", value=union_sub.nodes[edit_node].get('etype',''), key=f"new_type_{edit_node}")
                        col_a, col_b = st.columns(2)
                        with col_a:
                            if st.button("Apply edit"):
                                target_name = new_name.strip() if new_name.strip() else edit_node
                                if target_name != edit_node:
                                    if target_name in G_full:
                                        G_full = merge_nodes(G_full, edit_node, target_name)
                                        if push_to_neo4j and driver:
                                            neo4j_merge_rename(driver, edit_node, target_name)
                                    else:
                                        G_full.add_node(target_name)
                                        G_full = merge_nodes(G_full, edit_node, target_name)
                                        if push_to_neo4j and driver:
                                            neo4j_merge_rename(driver, edit_node, target_name)
                                
                                if target_name in G_full:
                                    if new_type.strip():
                                        G_full.nodes[target_name]['etype'] = new_type.strip()
                                else:
                                    if new_type.strip():
                                        G_full.nodes[edit_node]['etype'] = new_type.strip()
                                st.success("Edit applied")
                                
                                final_type = G_full.nodes[target_name].get('etype', '')
                                current_entities = st.session_state.get('kg_entities', [])
                                new_entities = [(target_name, final_type) if text == edit_node else (text, label) for text, label in current_entities]
                                st.session_state['kg_entities'] = list(dict.fromkeys(new_entities))

                                st.session_state['kg_graph'] = G_full
                                st.session_state['kg_nodes'] = sorted(list(G.nodes()))
                                st.session_state['kg_embeddings'] = embed_nodes(st.session_state['kg_nodes'])
                                st.session_state['subgraph'] = generate_union_subgraph(G_full, top_nodes, hops=hop_radius)
                                st.rerun() 
                        with col_b:
                            merge_target = st.selectbox("Merge into existing node (optional)", ["-"] + sorted([n for n in G_full.nodes() if n!=edit_node]), key=f"merge_target_{edit_node}")
                            if st.button("Merge selected into target") and merge_target and merge_target != "-":
                                if merge_target in G_full:
                                    G_full = merge_nodes(G_full, edit_node, merge_target)
                                    if push_to_neo4j and driver:
                                        neo4j_merge_rename(driver, edit_node, merge_target)
                                    st.success(f"Merged {edit_node} -> {merge_target}")
                                    
                                    target_etype = G_full.nodes[merge_target].get('etype', '')
                                    current_entities = st.session_state.get('kg_entities', [])
                                    new_entities = [(merge_target, target_etype) if text == edit_node else (text, label) for text, label in current_entities]
                                    st.session_state['kg_entities'] = list(dict.fromkeys(new_entities))
                                    
                                    st.session_state['kg_graph'] = G_full
                                    st.session_state['kg_nodes'] = sorted(list(G.nodes()))
                                    st.session_state['kg_embeddings'] = embed_nodes(st.session_state['kg_nodes'])
                                    st.session_state['subgraph'] = generate_union_subgraph(G_full, top_nodes, hops=hop_radius)
                                    st.session_state['search_results'] = [] 
                                    st.rerun() 

# ---------------- FEEDBACK (standalone view) ----------------
if menu == "Feedback":
    # ... (This section is unchanged) ...
    st.header("Feedback")
    if is_admin:
        st.subheader("Feedback dashboard")
        fb_df = fetch_feedback(1000)
        if fb_df.empty:
            st.info("No feedback yet")
        else:
            fb_df['time'] = fb_df['timestamp'].apply(lambda x: datetime.datetime.fromtimestamp(int(x)).strftime('%Y-%m-%d %H:%M'))
            st.dataframe(fb_df[['time','username','rating','comment']])
        st.markdown("---")
    
    st.subheader("Leave feedback (general)")
    qtxt = st.text_input("What did you search or attempt?")
    nodes_text = st.text_input("Top nodes (comma separated)")
    r = st.slider("Rate overall usefulness (1-5)", 1, 5, 4)
    c = st.text_area("Comment (optional)")
    if st.button("Submit general feedback"):
        top_nodes = [n.strip() for n in nodes_text.split(',')] if nodes_text else []
        try:
            add_feedback(username, qtxt or "", top_nodes, r, c)
            st.success("Feedback saved — thanks!")
        except Exception as e:
            st.error(f"Failed to save feedback: {e}")

# ---------------- SETTINGS ----------------
if menu == "Settings":
    # ... (This section is unchanged) ...
    st.header("Settings & Profile")
    st.subheader("Profile")
    new_display = st.text_input("Display name", value=display_name)
    
    st.subheader("Theme")
    st.caption("To change your theme (light/dark), please use the 'Settings' option in the `...` menu at the top-right of this page. Your choice will be saved in your browser.")
    
    if st.button("Update profile"):
        try:
            update_profile(username, display_name=new_display)
            st.session_state["display_name"] = new_display
            st.success("Profile updated")
            st.rerun() 
        except Exception as e:
            st.error(f"Failed to update profile: {e}")

    st.markdown("---")
    st.subheader("Account actions")
    if st.button("Generate deployment files (Dockerfile + compose) "):
        dep_path = write_deployment_files()
        st.success(f"Written to ./{dep_path}")
        
        try:
            files_to_zip = [
                os.path.join(dep_path, "Dockerfile"),
                os.path.join(dep_path, "docker-compose.yml"),
                os.path.join(dep_path, "requirements.txt"),
                os.path.join(dep_path, "README.md")
            ]
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w") as zf:
                for f in files_to_zip:
                    if os.path.exists(f):
                        zf.write(f, arcname=os.path.basename(f))
            zip_buf.seek(0)
            st.download_button("⬇️ Download deployment bundle", zip_buf, file_name="deployment_files.zip")
        except Exception as e:
            st.error(f"Could not create zip file: {e}")

# --- 5. SCRIPT EXECUTION ---
# Load models once
load_spacy_model()
load_embed_model()
