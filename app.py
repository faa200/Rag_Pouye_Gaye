import os
import tempfile
import streamlit as st
import pandas as pd
import yaml
from datetime import datetime
import importlib
import sqlite3
import csv

# --- Lecture du fichier de config ---
def read_config(file_path):
    with open(file_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
            return config
        except yaml.YAMLError as e:
            print(f"Error reading YAML file: {e}")
            return None

config = read_config("secrets/config.yaml")

# --- Mapping des frameworks ---
FRAMEWORKS = {
    "LangChain": "rag.langchain",
    "LlamaIndex": "rag.llamaindex"
}

st.set_page_config(
    page_title="Analyse de documents",
    page_icon="📄",
    layout="wide"
)

# --- CSS custom pour embellir ---
st.markdown("""
    <style>
    .stButton>button {background-color: #4CAF50; color: white; font-weight:bold;}
    .stRadio>div>label {font-weight: bold;}
    .stDownloadButton>button {background-color: #2196F3; color: white;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

if 'stored_files' not in st.session_state:
    st.session_state['stored_files'] = []

if 'framework' not in st.session_state:
    st.session_state['framework'] = "LangChain"

if 'k' not in st.session_state:
    st.session_state['k'] = 5

# --- Initialisation de la base de données ---
def init_db(db_path="feedbacks.db"):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS feedbacks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            note TEXT,
            question TEXT,
            response TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

# --- Créer le dossier d'export si besoin (corrigé) ---
def ensure_export_folder(folder="exports"):
    os.makedirs(folder, exist_ok=True)

# --- Sauvegarde du feedback dans la base ---
def save_feedback(note, question, response, db_path="feedbacks.db"):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("""
        INSERT INTO feedbacks (note, question, response)
        VALUES (?, ?, ?)
    """, (note, question, response))
    conn.commit()
    conn.close()

# --- Export du feedback dans un fichier CSV ---
def export_feedback_to_csv(note, question, response, folder="exports"):
    file_path = os.path.join(folder, "feedbacks.csv")
    file_exists = os.path.isfile(file_path)
    with open(file_path, mode='a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(["note", "question", "response", "created_at"])
        writer.writerow([note, question, response, datetime.now().isoformat()])

# --- Export de toute la base dans un fichier Excel ---
def export_feedbacks_to_excel(db_path="feedbacks.db", folder="exports"):
    file_path = os.path.join(folder, "feedbacks.xlsx")
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM feedbacks ORDER BY created_at DESC", conn)
    conn.close()
    df.to_excel(file_path, index=False)

def main():
    init_db()
    ensure_export_folder()

    # --- HEADER ---
    st.markdown("""
    # 📄 Analyse de documents avec IA
    *Chargez vos fichiers PDF, posez vos questions, et obtenez des réponses instantanées !*
    """)
    st.info("💡 **Astuce :** Vous pouvez charger plusieurs fichiers PDF à la fois. Sélectionnez le framework et la langue avant de commencer.")

    # --- SIDEBAR ---
    with st.sidebar:
        st.header("⚙️ Paramètres")
        selected_framework = st.radio(
            "Framework d'indexation",
            options=list(FRAMEWORKS.keys()),
            index=list(FRAMEWORKS.keys()).index(st.session_state['framework'])
        )
        langue = st.selectbox(
            "Langue de réponse",
            options=["Français", "Anglais", "Espagnol", "Allemand"],
            index=0
        )
        st.markdown("---")
        

    if st.session_state['framework'] != selected_framework:
        st.session_state['stored_files'] = []
        st.session_state['framework'] = selected_framework
        st.success("✅ Framework changé. Veuillez recharger vos documents.")

    framework_module = importlib.import_module(FRAMEWORKS[selected_framework])

    # --- MAIN LAYOUT ---
    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_files = st.file_uploader(
            label="📤 Déposez vos fichiers ici ou chargez-les",
            type=["pdf"],
            accept_multiple_files=True,
            key="fileUploader"
        )

        file_info = []
        if uploaded_files:
            for f in uploaded_files:
                size_in_kb = len(f.getvalue()) / 1024
                file_info.append({
                    "Nom du fichier": f.name,
                    "Taille (KB)": f"{size_in_kb:.2f}"
                })

                if f.name.endswith('.pdf') and f.name not in st.session_state['stored_files']:
                    temp_dir = tempfile.mkdtemp()
                    path = os.path.join(temp_dir, "temp.pdf")
                    with open(path, "wb") as outfile:
                        outfile.write(f.getvalue())
                    framework_module.store_pdf_file(path, f.name)
                    st.session_state['stored_files'].append(f.name)

            df = pd.DataFrame(file_info)
            st.markdown("### 📚 Fichiers chargés")
            st.table(df)

        # Gestion suppression fichiers
        current_files = {f['Nom du fichier'] for f in file_info}
        files_to_be_deleted = set(st.session_state['stored_files']) - current_files
        for name in files_to_be_deleted:
            st.session_state['stored_files'].remove(name)
            framework_module.delete_file_from_store(name)

        # --- Choix du nombre de documents similaires (k) ---
        st.markdown("### 🔢 Nombre de documents similaires à récupérer")
        k = st.slider(
            "Sélectionnez combien de documents similaires doivent être pris en compte dans l'analyse ",
            min_value=1,
            max_value=20,
            value=st.session_state['k'],
            step=1,
            key="slider_k"
        )
        st.session_state['k'] = k

        # --- Question utilisateur ---
        st.markdown("### ❓ Posez votre question")
        question = st.text_input("Votre question ici")

        response = ""
        if st.button("🚀 Analyser la question"):
            if not question.strip():
                st.warning("Veuillez saisir une question.")
            else:
                try:
                    response = framework_module.answer_question(question, langue, k)
                except Exception as e:
                    st.error("Erreur de connexion à l'API. Vérifiez votre connexion Internet et votre clé API.")
                    response = ""
                if response:
                    st.success("✅ **Réponse générée :**")
                    st.write(response)
        else:
            st.markdown("📝 La réponse s'affichera ici après l'analyse.")

        # --- Feedback utilisateur ---
        if response:
            st.markdown("### ✨ Votre avis sur la réponse")
            feedback = st.radio(
                "Comment évaluez-vous la qualité de la réponse ?",
                options=["👍 Très bien", "👎 Mauvais", "🤔 Moyenne"],
                horizontal=True,
                key="user_feedback"
            )
            if st.button("💬 Envoyer le feedback"):
                if feedback:
                    save_feedback(feedback, question, response)
                    export_feedback_to_csv(feedback, question, response)
                    export_feedbacks_to_excel()  # Export Excel automatique
                    st.success("🙏 Merci pour votre feedback !")

    with col2:
        st.markdown("### 📥 Feedbacks")
        # Affichage des feedbacks enregistrés (optionnel, pour admin)
        if st.checkbox("Afficher les feedbacks enregistrés"):
            conn = sqlite3.connect("feedbacks.db")
            df_fb = pd.read_sql_query("SELECT * FROM feedbacks ORDER BY created_at DESC", conn)
            st.dataframe(df_fb)
            conn.close()

        # Bouton de téléchargement du CSV
        export_path = os.path.join("exports", "feedbacks.csv")
        if os.path.exists(export_path):
            with open(export_path, "rb") as f:
                st.download_button(
                    label="⬇️ Télécharger les feedbacks (CSV)",
                    data=f,
                    file_name="feedbacks.csv",
                    mime="text/csv"
                )

        # Bouton de téléchargement du Excel
        export_excel_path = os.path.join("exports", "feedbacks.xlsx")
        if os.path.exists(export_excel_path):
            with open(export_excel_path, "rb") as f:
                st.download_button(
                    label="⬇️ Télécharger les feedbacks (Excel)",
                    data=f,
                    file_name="feedbacks.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

    # --- FOOTER ---
    st.markdown("""
    <hr>
    <div style='text-align:center; color:gray'>
    
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

      