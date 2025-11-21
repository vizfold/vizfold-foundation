#!/bin/bash

# Startup script for OpenFold Attention Visualization Webapp
# Starts both backend and frontend servers

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo ""
echo "========================================="
echo " OpenFold Attention Visualization"
echo " Starting Backend + Frontend"
echo "========================================="
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if backend directory exists
if [ ! -d "backend" ]; then
    echo -e "${RED}Error: backend directory not found${NC}"
    exit 1
fi

# Check if frontend directory exists
if [ ! -d "frontend" ]; then
    echo -e "${RED}Error: frontend directory not found${NC}"
    exit 1
fi

# Start backend
echo -e "${YELLOW}Starting backend on port 8000...${NC}"
cd backend

# Use SQLite for simplicity (no PostgreSQL setup needed)
export DATABASE_URL="sqlite:///./openfold_viz.db"
export REDIS_URL="redis://localhost:6379/0"

# Check if uvicorn is available
if ! command -v uvicorn &> /dev/null; then
    echo -e "${RED}Error: uvicorn not found. Install with: pip install uvicorn${NC}"
    exit 1
fi

# Start backend in background
uvicorn app.main:app --reload --port 8000 > ../backend.log 2>&1 &
BACKEND_PID=$!

echo -e "${GREEN}Backend started (PID: $BACKEND_PID)${NC}"
echo "  Logs: $(pwd)/../backend.log"

# Wait for backend to be ready
echo -n "  Waiting for backend to start"
for i in {1..30}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo -e " ${GREEN}✓${NC}"
        break
    fi
    echo -n "."
    sleep 1

    if [ $i -eq 30 ]; then
        echo -e " ${RED}✗${NC}"
        echo -e "${RED}Backend failed to start. Check backend.log for errors${NC}"
        kill $BACKEND_PID 2>/dev/null
        exit 1
    fi
done

# Start frontend
cd ../frontend
echo ""
echo -e "${YELLOW}Starting frontend...${NC}"

# Check if npm is available
if ! command -v npm &> /dev/null; then
    echo -e "${RED}Error: npm not found. Install Node.js first${NC}"
    kill $BACKEND_PID
    exit 1
fi

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo -e "${YELLOW}Installing frontend dependencies...${NC}"
    npm install
fi

# Start frontend in background
npm run dev > ../frontend.log 2>&1 &
FRONTEND_PID=$!

echo -e "${GREEN}Frontend started (PID: $FRONTEND_PID)${NC}"
echo "  Logs: $(pwd)/../frontend.log"

# Wait a moment for frontend to start
sleep 3

echo ""
echo "========================================="
echo -e "${GREEN}✓ Both servers are running!${NC}"
echo "========================================="
echo ""
echo "  Backend:  http://localhost:8000"
echo "  API Docs: http://localhost:8000/docs"
echo "  Frontend: http://localhost:5173"
echo "            (or check frontend.log for actual port)"
echo ""
echo "  Backend logs:  tail -f backend.log"
echo "  Frontend logs: tail -f frontend.log"
echo ""
echo "Press Ctrl+C to stop both servers"
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo -e "${YELLOW}Stopping servers...${NC}"
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    echo -e "${GREEN}Servers stopped${NC}"
    exit 0
}

# Trap Ctrl+C
trap cleanup INT TERM

# Wait for either process to exit
wait $BACKEND_PID $FRONTEND_PID
