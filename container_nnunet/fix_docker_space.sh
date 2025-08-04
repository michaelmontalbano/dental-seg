#!/bin/bash

echo "🧹 Cleaning up Docker to free space..."

# Remove unused containers
echo "📦 Removing stopped containers..."
docker container prune -f

# Remove unused images
echo "🖼️ Removing dangling images..."
docker image prune -f

# Remove all unused images (not just dangling)
echo "🖼️ Removing unused images..."
docker image prune -a -f

# Remove unused volumes
echo "💾 Removing unused volumes..."
docker volume prune -f

# Remove unused networks
echo "🌐 Removing unused networks..."
docker network prune -f

# Remove build cache
echo "🗑️ Removing build cache..."
docker builder prune -f

# Show disk usage
echo ""
echo "📊 Current Docker disk usage:"
docker system df

echo ""
echo "✅ Cleanup complete!"
echo ""
echo "💡 If you still need more space, you can:"
echo "   1. Increase Docker Desktop disk allocation in Settings > Resources"
echo "   2. Run: docker system prune -a --volumes -f (removes ALL unused data)"
