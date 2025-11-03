
import re
import tempfile
import json
import streamlit as st
import networkx as nx
from pyvis.network import Network
from neo4j import GraphDatabase
import torch
from sentence_transformers import SentenceTransformer, util
import spacy
from spacy.lang.en.stop_words import STOP_WORDS # <-- NEW IMPORT
from typing import List, Tuple, Dict

from config import NEO4J_URI, NEO4J_USER

# ---------------- Cached Models ----------------
@st.cache_resource
def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except Exception:
        # fallback - will raise during runtime if not installed
        return spacy.load("en_core_web_sm")

nlp = load_spacy_model()

@st.cache_resource
def load_embed_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embed_model = load_embed_model()

# --- NEW: Preprocessing Function ---
def preprocess_text(text: str) -> str:
    """Cleans raw text for better NLP extraction."""
    if not isinstance(text, str):
        return ""
    
    # 1. Lowercase
    text = text.lower()
    # 2. Remove punctuation
    text = re.sub(r"[^\w\s]", "", text)
    
    # 3. Remove stop words
    # We re-use the global 'nlp' model
    doc = nlp(text)
    clean_tokens = [token.text for token in doc if not token.is_stop]
    
    return " ".join(clean_tokens)
# --- End of new function ---

# ---------------- Utilities: Extraction ----------------
def extract_entities_and_relations(text: str) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str, str]]]:
    # ... (This function is unchanged) ...
    if not isinstance(text, str) or not text.strip():
        return [], []

    doc = nlp(text)
    entities = [(ent.text.strip(), ent.label_) for ent in doc.ents]

    triples = []
    dev_pat = r"(?P<sub>[A-Z][\w\s\.#]+?)\s*(?:developed|founded|created|established|invented|built)\s*(?P<obj>[\w\s\-/&]+?)(?: in (?P<year>\d{4}))?(?:[.,]|$)"
    for m in re.finditer(dev_pat, text, flags=re.I):
        s = m.group("sub").strip()
        o = m.group("obj").strip()
        y = m.group("year")
        if s and o:
            triples.append((s, "DEVELOPED", o))
        if s and y:
            triples.append((s, "DEVELOPED_IN", y))

    intro = re.search(r"i\s*'?m\s+(?P<name>[A-Z]?[a-zA-Z]+(?:\s+[A-Za-z]+)?)\s+from\s+(?P<city>[A-Za-z\s]+)\s+working\s+in\s+(?P<org>[A-Za-z0-9 &\.]+)(?:\s+since\s+(?P<year>\d{4}))?", text, flags=re.I)
    if intro:
        n = intro.group("name").title()
        c = intro.group("city").title()
        o = intro.group("org").strip()
        y = intro.group("year")
        triples.append((n, "BASED_IN", c))
        triples.append((n, "EMPLOYED_AT", o))
        if y:
            triples.append((n, "EMPLOYED_SINCE", y))

    for sent in doc.sents:
        persons = [ent.text for ent in sent.ents if ent.label_ == "PERSON"]
        orgs = [ent.text for ent in sent.ents if ent.label_ == "ORG"]
        gpes = [ent.text for ent in sent.ents if ent.label_ == "GPE"]
        for p in persons:
            for o in orgs:
                triples.append((p, "WORKS_AT", o))
            for g in gpes:
                triples.append((p, "LIVES_IN", g))

    unique_triples = list(dict.fromkeys(triples))
    unique_entities = list(dict.fromkeys(entities))

    return unique_entities, unique_triples

# ---------------- Neo4j helpers ----------------
# ... (All Neo4j functions are unchanged) ...
@st.cache_resource
def create_neo4j_driver(password: str):
    if not password:
        return None
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, password))
        driver.verify_connectivity()
        return driver
    except Exception as e:
        st.warning(f"Neo4j connection failed: {e}")
        return None

def neo4j_count_nodes_and_rels(driver):
    if not driver:
        return None
    try:
        with driver.session() as session:
            node_count = session.run("MATCH (n) RETURN count(n) AS c").single().get("c")
            rel_count = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single().get("c")
            labels = {}
            try:
                res = session.run("MATCH (n) RETURN labels(n) AS labs, count(*) AS c")
                for r in res:
                    labs = r["labs"] or []
                    for lab in labs:
                        labels[lab] = labels.get(lab, 0) + r["c"]
            except Exception:
                labels = {}
            return {"nodes": int(node_count), "rels": int(rel_count), "labels": labels}
    except Exception as e:
        st.warning(f"Neo4j metric query failed: {e}")
        return None

def neo4j_merge_rename(driver, old_name, new_name):
    if not driver:
        return False
    try:
        with driver.session() as session:
            query = """
            MATCH (a:Entity {name: $old})
            MERGE (b:Entity {name: $new})
            WITH a,b
            CALL apoc.refactor.mergeNodes([a,b]) YIELD node RETURN node
            """
            try:
                session.run(query, old=old_name, new=new_name)
                return True
            except Exception:
                session.run("MERGE (b:Entity {name: $new})", new=new_name)
                session.run("MATCH (x)-[r]->(a {name:$old}) CREATE (x)-[r2:TYPE]->(b) DELETE r", old=old_name, new=new_name)
                session.run("MATCH (a {name:$old})-[r]->(x) CREATE (b)-[r2:TYPE]->(x) DELETE r", old=old_name, new=new_name)
                session.run("MATCH (a {name:$old}) DELETE a", old=old_name)
                return True
    except Exception as e:
        st.warning(f"Neo4j rename/merge failed: {e}")
        return False

def neo4j_add_relation(driver, s, rel, o):
    if not driver:
        return False
    try:
        with driver.session() as session:
            lbl = re.sub(r"[^A-Z0-9_]", "_", rel.upper())
            query = f"MERGE (a:Entity {{name:$s}}) MERGE (b:Entity {{name:$o}}) MERGE (a)-[:{lbl}]->(b)"
            session.run(query, s=s, o=o)
            return True
    except Exception as e:
        st.warning(f"Neo4j add relation failed: {e}")
        return False

# ---------------- NetworkX helpers ----------------
# ... (All NetworkX functions are unchanged) ...
def build_networkx_graph(triples: List[Tuple[str,str,str]]) -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()
    for s, r, o in triples:
        G.add_node(s)
        G.add_node(o)
        G.add_edge(s, o, relation=r)
    return G

def annotate_nodes_with_entities(G: nx.Graph, entities: List[Tuple[str,str]]):
    ent_map = {text.strip(): label for text, label in entities}
    for n in list(G.nodes()):
        if n in ent_map:
            G.nodes[n]['etype'] = ent_map[n]
        else:
            for t, lab in ent_map.items():
                if t.lower() == n.lower():
                    G.nodes[n]['etype'] = lab
                    break

def embed_nodes(nodes: List[str]):
    if not nodes:
        return None
    return embed_model.encode(nodes, convert_to_tensor=True)

def semantic_query_topk(query: str, nodes: List[str], node_embeds, k: int = 5) -> List[Dict]:
    if not query.strip() or not nodes:
        return []
    q_emb = embed_model.encode(query, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(q_emb, node_embeds)[0]
    topk = torch.topk(cos_scores, k=min(k, len(nodes)))
    results = []
    for score, idx in zip(topk[0], topk[1]):
        results.append({"node": nodes[int(idx.item())], "score": float(score.item())})
    return results

def generate_union_subgraph(G: nx.Graph, seeds: List[str], hops: int = 1) -> nx.Graph:
    ego_list = []
    for s in seeds:
        if s in G:
            ego_list.append(nx.ego_graph(G, s, radius=hops))
    if not ego_list:
        return nx.Graph()
    return nx.compose_all(ego_list)

def create_pyvis_html(G: nx.Graph, highlight: List[str] = None, show_relation_labels: bool = True) -> str:
    if highlight is None:
        highlight = []
    net = Network(height="650px", width="100%", bgcolor="#FFFFFF", font_color="#111", directed=True)
    try:
        net.barnes_hut(spring_length=200, gravity=-30000, central_gravity=0.2)
    except Exception:
        pass
    for n, data in G.nodes(data=True):
        title = n
        etype = data.get("etype", "")
        if etype:
            title += f" ({etype})"
        color = "#FF6B6B" if n in highlight else "#7EC8E3"
        size = 30 if n in highlight else 16
        net.add_node(n, label=n, title=title, size=size, color=color)
    for u, v, data in G.edges(data=True):
        rel = data.get("relation", "")
        if show_relation_labels:
            net.add_edge(u, v, title=rel, label=rel)
        else:
            net.add_edge(u, v, title=rel)
    tmp = tempfile.NamedTemporaryFile(suffix=".html", delete=False)
    net.save_graph(tmp.name)
    return tmp.name

def export_graph_to_json(G: nx.Graph) -> str:
    data = nx.readwrite.json_graph.node_link_data(G)
    return json.dumps(data, indent=2)

def merge_nodes(G: nx.Graph, source: str, target: str) -> nx.Graph:
    if source not in G or target not in G:
        return G
    for u, v, key, data in list(G.edges(keys=True, data=True)):
        if u == source:
            G.add_edge(target, v, **data)
        if v == source:
            G.add_edge(u, target, **data)
    if 'etype' in G.nodes[source] and 'etype' not in G.nodes[target]:
        G.nodes[target]['etype'] = G.nodes[source]['etype']
    if source in G:
        G.remove_node(source)
    return G
