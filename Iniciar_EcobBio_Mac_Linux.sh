#!/bin/bash
# ============================================================
#   EcoBio - Clasificador de Fauna Uruguaya (macOS/Linux)
# ============================================================

echo "============================================================"
echo "  EcoBio - Clasificador de Fauna Uruguaya"
echo "============================================================"
echo ""

# Ir al directorio donde está este script
cd "$(dirname "$0")"

# ── Detectar Python ───────────────────────────────────────────
if ! command -v python3 &>/dev/null; then
    echo "[ERROR] Python no encontrado."
    echo "        Instalalo desde: https://www.python.org/downloads/"
    echo "        O usando Homebrew: brew install python"
    read -p "Presioná Enter para salir..."
    exit 1
fi

# ═════════════════════════════════════════════════════════════
# CASO A: ya existe venv/ lista (pre-instalada o ya instalada antes)
# ═════════════════════════════════════════════════════════════
if [ -f "venv/installed.flag" ]; then
    echo "[OK] Entorno virtual encontrado. Saltando instalación."
else
    # ═════════════════════════════════════════════════════════
    # CASO B: no hay venv, hay que crearlo e instalar todo
    # ═════════════════════════════════════════════════════════
    echo "[1/4] Creando entorno virtual..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "[ERROR] No se pudo crear el entorno virtual."
        read -p "Presioná Enter para salir..."
        exit 1
    fi

    source venv/bin/activate

    echo "[2/4] Actualizando pip..."
    pip install --upgrade pip --quiet

    echo "[3/4] Instalando dependencias base..."
    pip install -r requirements.txt --quiet
    if [ $? -ne 0 ]; then
        echo "[ERROR] Falló la instalación de dependencias base."
        echo "        Verificá tu conexión a internet e intentá de nuevo."
        read -p "Presioná Enter para salir..."
        exit 1
    fi

    # ── Instalar megadetector ─────────────────────────────────
    echo "[4/4] Instalando megadetector y speciesnet..."

    if [ -d "packages/megadetector" ]; then
        echo "     - megadetector: instalando desde carpeta local..."
        pip install packages/megadetector/ --quiet
    else
        echo "     - megadetector: descargando desde GitHub..."
        pip install git+https://github.com/agentmorris/MegaDetector.git --quiet
    fi
    if [ $? -ne 0 ]; then
        echo "[ERROR] Falló la instalación de megadetector."
        read -p "Presioná Enter para salir..."
        exit 1
    fi

    if [ -d "packages/cameratrapai" ]; then
        echo "     - speciesnet: instalando desde carpeta local..."
        pip install packages/cameratrapai/ --quiet
    else
        echo "     - speciesnet: descargando desde GitHub..."
        pip install git+https://github.com/google/cameratrapai.git --quiet
    fi
    if [ $? -ne 0 ]; then
        echo "[ERROR] Falló la instalación de speciesnet."
        read -p "Presioná Enter para salir..."
        exit 1
    fi

    # Marcar como instalado para no repetir el proceso
    touch venv/installed.flag
    echo ""
    echo "[OK] Instalación completada exitosamente."
    echo ""
fi

# ═════════════════════════════════════════════════════════════
# Activar entorno y lanzar Streamlit
# ═════════════════════════════════════════════════════════════
source venv/bin/activate

echo " Iniciando aplicación..."
echo " Abriendo en el navegador: http://localhost:8501"
echo " Para cerrar la app, presioná Ctrl+C en esta terminal."
echo ""

# Abrir el navegador luego de 4 segundos (en background)
(sleep 4 && open http://localhost:8501) &

streamlit run app.py --server.port 8501 --server.headless true --browser.gatherUsageStats false
