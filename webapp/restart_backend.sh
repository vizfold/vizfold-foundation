#!/bin/bash

# Quick restart script for backend after applying fixes

echo ""
echo "========================================="
echo " Restarting Backend with Fixes Applied"
echo "========================================="
echo ""

# Kill existing backend on port 8000
echo "→ Stopping existing backend..."
if lsof -ti:8000 > /dev/null 2>&1; then
    lsof -ti:8000 | xargs kill -9 2>/dev/null
    echo "✓ Stopped backend on port 8000"
else
    echo "  (No backend running on port 8000)"
fi

# Wait a moment
sleep 1

# Navigate to backend directory
cd "$(dirname "$0")/backend"

echo ""
echo "→ Starting backend with SQLite..."
echo "  Database: sqlite:///./openfold_viz.db"
echo "  Port: 8000"
echo ""

# Start backend with SQLite
DATABASE_URL="sqlite:///./openfold_viz.db" uvicorn app.main:app --reload --port 8000 &
BACKEND_PID=$!

# Wait for backend to start
echo -n "→ Waiting for backend"
for i in {1..10}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo " ✓"
        break
    fi
    echo -n "."
    sleep 1
done

echo ""
echo "========================================="
echo "✅ Backend is running!"
echo "========================================="
echo ""
echo "  URL: http://localhost:8000"
echo "  API Docs: http://localhost:8000/docs"
echo "  Health: http://localhost:8000/health"
echo ""
echo "Test it:"
echo "  curl http://localhost:8000/health"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Keep script running
wait $BACKEND_PID
