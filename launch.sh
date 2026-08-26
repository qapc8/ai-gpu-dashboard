#!/bin/bash
# AI GPU Market Dashboard Launcher
#
# Usage:
#   ./launch.sh            — Serve the dashboard at http://localhost:8050
#   ./launch.sh refresh    — Run the data pipeline, then serve
#   ./launch.sh install    — Install dependencies
#
# The terminal and AI-only modes are gone: terminal_dashboard.py and
# ai_analyzer.py were deleted. Both had already stopped working -- they
# imported get_utilization_summary, retired when per-provider utilization was
# dropped for being unsourceable -- so neither had run in some time.

set -e
cd "$(dirname "$0")"

GREEN='\033[0;32m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

install_deps() {
    echo -e "${CYAN}${BOLD}Installing dependencies...${NC}"
    pip3 install -r requirements.txt
    echo -e "${GREEN}Done!${NC}"
}

refresh_data() {
    echo -e "${CYAN}${BOLD}Refreshing data (this takes a few minutes)...${NC}"
    python3 scripts/update_data.py
}

launch_web() {
    echo -e "${CYAN}${BOLD}Serving dashboard...${NC}"
    echo -e "${GREEN}Open http://localhost:8050 in your browser${NC}"
    python3 server.py
}

case "${1:-web}" in
    install)
        install_deps
        ;;
    refresh|update)
        refresh_data
        launch_web
        ;;
    web|w|browser|both|all)
        launch_web
        ;;
    *)
        echo "Usage: $0 {web|refresh|install}"
        ;;
esac
