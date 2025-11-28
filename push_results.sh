#!/bin/bash
# Push training results and visualizations to GitHub after training completes

echo "================================"
echo "Pushing results to GitHub"
echo "================================"

# Check if results and visualizations exist
if [ ! -d "results" ] && [ ! -d "visualizations" ]; then
    echo "No results or visualizations found to push."
    exit 1
fi

# Stage results and visualizations
echo "Staging results/ and visualizations/ directories..."
git add results/ visualizations/

# Check if there are changes to commit
if git diff --cached --quiet; then
    echo "No changes to commit."
    exit 0
fi

# Create commit with results
echo "Creating commit..."
git commit -m "Add training results and visualizations from Colab A100 training"

# Push to GitHub
echo "Pushing to GitHub..."
git push origin main

echo "================================"
echo "Results pushed successfully!"
echo "================================"
