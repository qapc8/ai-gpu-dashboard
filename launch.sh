#!/bin/bash
# AI GPU Market Terminal Dashboard Launcher
# Usage:
#   ./launch.sh terminal   — Launch terminal dashboard
#   ./launch.sh web        — Launch web dashboard (browser)
#   ./launch.sh both       — Launch both
#   ./launch.sh install    — Install dependencies

set -e
cd "$(dirname "$0")"

RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

install_deps() {
    echo -e "${CYAN}${BOLD}Installing dependencies...${NC}"
    pip3 install -r requirements.txt
    echo -e "${GREEN}Done!${NC}"
}

launch_terminal() {
    echo -e "${CYAN}${BOLD}Launching Terminal Dashboard...${NC}"
    python3 terminal_dashboard.py
}

launch_web() {
    echo -e "${CYAN}${BOLD}Launching Web Dashboard...${NC}"
    echo -e "${GREEN}Open http://localhost:8050 in your browser${NC}"
    python3 server.py
}

case "${1:-both}" in
    install)
        install_deps
        ;;
    terminal|term|t)
        launch_terminal
        ;;
    web|w|browser)
        launch_web
        ;;
    both|all)
        echo -e "${CYAN}${BOLD}=== AI GPU MARKET TERMINAL ===${NC}"
        echo ""
        echo -e "  ${GREEN}1)${NC} Terminal:  python3 terminal_dashboard.py"
        echo -e "  ${GREEN}2)${NC} Browser:   python3 server.py  →  http://localhost:8050"
        echo -e "  ${GREEN}3)${NC} AI Only:   python3 ai_analyzer.py"
        echo ""
        echo -e "${CYAN}Choose mode [1/2/3]:${NC} "
        read -r choice
        case "$choice" in
            1) launch_terminal ;;
            2) launch_web ;;
            3) python3 ai_analyzer.py ;;
            *) echo "Starting web dashboard..." && launch_web ;;
        esac
        ;;
    *)
        echo "Usage: $0 {terminal|web|both|install}"
        ;;
esac
