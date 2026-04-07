#!/usr/bin/env bash
#
# package_dmg.sh — Build RFantibodyOptimizer.app with all dependencies
#                  bundled inside, then create a distributable DMG.
#
# Usage:  ./package_dmg.sh
#
# Requirements:
#   - Xcode CLI tools
#   - The repo symlinks (models/, pilot_mps/) must resolve
#   - The uv-managed Python at the path in pilot_mps/.venv/pyvenv.cfg
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
APP_NAME="RFantibodyOptimizer"
BUILD_DIR="/tmp/${APP_NAME}-release-build"
APP_PATH="${BUILD_DIR}/Build/Products/Release/${APP_NAME}.app"
RESOURCES="${APP_PATH}/Contents/Resources"
DMG_DIR="/tmp/${APP_NAME}-dmg"
DMG_OUT="${REPO_ROOT}/${APP_NAME}.dmg"

# Resolve symlinks to real paths
VENV_DIR="$(readlink -f "${REPO_ROOT}/pilot_mps/.venv")"
MODELS_DIR="$(readlink -f "${REPO_ROOT}/models")"
PYTHON_HOME="$(head -1 "${VENV_DIR}/pyvenv.cfg" | sed 's/home = //')"
PYTHON_BASE="$(dirname "${PYTHON_HOME}")"  # e.g. .../cpython-3.10-macos-aarch64-none

echo "==> Configuration"
echo "    Repo root:    ${REPO_ROOT}"
echo "    Venv:         ${VENV_DIR}"
echo "    Models:       ${MODELS_DIR}"
echo "    Python base:  ${PYTHON_BASE}"
echo ""

# ── Step 1: Build Release ────────────────────────────────────────────
echo "==> Building Release..."
xcodebuild -project "${REPO_ROOT}/${APP_NAME}.xcodeproj" \
    -scheme "${APP_NAME}" \
    -configuration Release \
    -derivedDataPath "${BUILD_DIR}" \
    build 2>&1 | grep -E "(BUILD|error:|warning:.*error)" || true

if [ ! -d "${APP_PATH}" ]; then
    echo "ERROR: Build failed — ${APP_PATH} not found"
    exit 1
fi
echo "    Built: ${APP_PATH}"
echo ""

# ── Step 2: Bundle Python runtime ────────────────────────────────────
echo "==> Bundling Python runtime..."
PYTHON_DEST="${RESOURCES}/python"
mkdir -p "${PYTHON_DEST}"

# Copy standalone Python installation (stdlib + binary)
rsync -a --exclude='__pycache__' --exclude='test/' --exclude='tests/' \
    --exclude='*.pyc' --exclude='idle*' --exclude='tkinter' \
    --exclude='turtle*' --exclude='turtledemo' \
    --exclude='itcl*' --exclude='libtcl*' --exclude='libtcl9tk*' \
    --exclude='tcl9*' --exclude='tk9*' --exclude='thread*' \
    "${PYTHON_BASE}/" "${PYTHON_DEST}/"

# Copy site-packages from venv
SITE_SRC="${VENV_DIR}/lib/python3.10/site-packages"
SITE_DST="${PYTHON_DEST}/lib/python3.10/site-packages"
echo "    Copying site-packages (this may take a moment)..."
rsync -a --exclude='__pycache__' --exclude='*.pyc' \
    --exclude='pip' --exclude='pip-*' \
    --exclude='setuptools' --exclude='setuptools-*' \
    --exclude='wheel' --exclude='wheel-*' \
    --exclude='_distutils_hack' \
    "${SITE_SRC}/" "${SITE_DST}/"

# Write a pyvenv.cfg-style marker so Python finds site-packages
# (We're using the standalone Python directly, not a venv, so site-packages
#  is already in the default lib path — no pyvenv.cfg needed.)

echo "    Python bundled: $(du -sh "${PYTHON_DEST}" | cut -f1)"
echo ""

# ── Step 3: Bundle Python source code ────────────────────────────────
echo "==> Bundling pipeline source..."
rsync -a --exclude='__pycache__' --exclude='*.pyc' \
    "${REPO_ROOT}/src/" "${RESOURCES}/src/"

rsync -a --exclude='__pycache__' --exclude='*.pyc' \
    "${REPO_ROOT}/include/SE3Transformer/" "${RESOURCES}/include/SE3Transformer/"

rsync -a --exclude='__pycache__' --exclude='*.pyc' \
    "${REPO_ROOT}/scripts/" "${RESOURCES}/scripts/"

echo "    Source bundled."
echo ""

# ── Step 4: Bundle model checkpoints ─────────────────────────────────
echo "==> Bundling model checkpoints..."
mkdir -p "${RESOURCES}/models"
for ckpt in RFdiffusion_Ab.pt ProteinMPNN_v48_noise_0.2.pt RF2_ab.pt; do
    if [ -f "${MODELS_DIR}/${ckpt}" ]; then
        cp "${MODELS_DIR}/${ckpt}" "${RESOURCES}/models/${ckpt}"
        echo "    Copied ${ckpt} ($(du -sh "${MODELS_DIR}/${ckpt}" | cut -f1))"
    else
        echo "    WARNING: ${ckpt} not found in ${MODELS_DIR}"
    fi
done
echo ""

# ── Step 5: Strip extended attributes & ad-hoc code sign ─────────────
# Finder/rsync can leave resource forks and xattrs that break code signing.
echo "==> Stripping extended attributes..."
xattr -cr "${APP_PATH}"

echo "==> Code signing..."
codesign --force --deep --sign - "${APP_PATH}"
echo "    Signed."
echo ""

# ── Step 6: Create DMG ───────────────────────────────────────────────
echo "==> Creating DMG..."
rm -rf "${DMG_DIR}"
mkdir -p "${DMG_DIR}"

# Use ditto instead of cp -R to avoid reintroducing resource forks
ditto "${APP_PATH}" "${DMG_DIR}/${APP_NAME}.app"
ln -s /Applications "${DMG_DIR}/Applications"

# Remove old DMG if present
rm -f "${DMG_OUT}"

hdiutil create -volname "${APP_NAME}" \
    -srcfolder "${DMG_DIR}" \
    -ov -format UDZO \
    "${DMG_OUT}"

echo ""
echo "==> Done!"
echo "    App:  ${APP_PATH}"
echo "    DMG:  ${DMG_OUT}"
echo "    Size: $(du -sh "${DMG_OUT}" | cut -f1)"
