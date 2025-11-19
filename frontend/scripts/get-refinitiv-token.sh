#!/bin/bash

# Refinitiv Token Helper Script
# This makes it easy to set a fresh token without copying/pasting long commands

echo "🔑 Refinitiv Token Setup"
echo "================================"
echo ""
echo "1. Go to: https://api.refinitiv.com"
echo "2. Sign in: anish@lithos-ai.com / 123@Ninja"
echo "3. Click any API endpoint"
echo "4. Look at the Python code sample"
echo "5. Copy the Bearer token"
echo ""
echo "Then paste it below:"
echo ""
read -p "Bearer Token: " TOKEN

if [ -z "$TOKEN" ]; then
    echo "❌ No token provided"
    exit 1
fi

# Save to .env file
echo "REFINITIV_BEARER_TOKEN=\"$TOKEN\"" > .env.refinitiv
echo ""
echo "✅ Token saved to .env.refinitiv"
echo ""
echo "Now run:"
echo "  source .env.refinitiv"
echo "  npx tsx scripts/refinitiv-comprehensive-extractor.ts"
