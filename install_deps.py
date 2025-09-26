import subprocess
import sys

def install_package(package):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ {package} installé avec succès")
    except subprocess.CalledProcessError as e:
        print(f"❌ Erreur lors de l'installation de {package}: {e}")

print("📦 Installation des dépendances...")

packages = [
    "langchain-neo4j",
    "streamlit"
]

for package in packages:
    install_package(package)

print("🎉 Installation terminée!")