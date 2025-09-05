#!/bin/bash

# Test 1: Basic embedding with default settings
echo "Test 1: Basic embedding (default settings)"
curl -s -X POST \
  -H "Authorization: Bearer $REPLICATE_API_TOKEN" \
  -H "Content-Type: application/json" \
  -H "Prefer: wait" \
  -d '{
    "version": "VERSION_HASH_HERE",
    "input": {
      "text": "Machine learning helps computers understand data patterns"
    }
  }' \
  https://api.replicate.com/v1/predictions

echo -e "\n\n"

# Test 2: Search query vs document embedding
echo "Test 2: Search query embedding"
curl -s -X POST \
  -H "Authorization: Bearer $REPLICATE_API_TOKEN" \
  -H "Content-Type: application/json" \
  -H "Prefer: wait" \
  -d '{
    "version": "VERSION_HASH_HERE", 
    "input": {
      "text": "What is Python programming language?",
      "task": "retrieval_query"
    }
  }' \
  https://api.replicate.com/v1/predictions

echo -e "\n\n"

# Test 3: Document embedding for search
echo "Test 3: Document embedding for search"
curl -s -X POST \
  -H "Authorization: Bearer $REPLICATE_API_TOKEN" \
  -H "Content-Type: application/json" \
  -H "Prefer: wait" \
  -d '{
    "version": "VERSION_HASH_HERE",
    "input": {
      "text": "Python is a high-level programming language used for web development, data science, and automation",
      "task": "retrieval_document"
    }
  }' \
  https://api.replicate.com/v1/predictions

echo -e "\n\n"

# Test 4: Base64 output format
echo "Test 4: Base64 output format"
curl -s -X POST \
  -H "Authorization: Bearer $REPLICATE_API_TOKEN" \
  -H "Content-Type: application/json" \
  -H "Prefer: wait" \
  -d '{
    "version": "VERSION_HASH_HERE",
    "input": {
      "text": "Testing base64 embedding output format",
      "output_format": "base64"
    }
  }' \
  https://api.replicate.com/v1/predictions

echo -e "\n\n"

# Test 5: Classification task
echo "Test 5: Classification task"
curl -s -X POST \
  -H "Authorization: Bearer $REPLICATE_API_TOKEN" \
  -H "Content-Type: application/json" \
  -H "Prefer: wait" \
  -d '{
    "version": "VERSION_HASH_HERE",
    "input": {
      "text": "This movie was absolutely fantastic! Great acting and amazing story.",
      "task": "classification",
      "normalize": false
    }
  }' \
  https://api.replicate.com/v1/predictions

echo -e "\n\n"

# Test 6: Code retrieval task
echo "Test 6: Code retrieval task"
curl -s -X POST \
  -H "Authorization: Bearer $REPLICATE_API_TOKEN" \
  -H "Content-Type: application/json" \
  -H "Prefer: wait" \
  -d '{
    "version": "VERSION_HASH_HERE",
    "input": {
      "text": "def quicksort(arr): return [] if not arr else quicksort([x for x in arr[1:] if x < arr[0]]) + [arr[0]] + quicksort([x for x in arr[1:] if x >= arr[0]])",
      "task": "code_retrieval"
    }
  }' \
  https://api.replicate.com/v1/predictions

echo -e "\n\n"

# Test 7: Error handling - empty text
echo "Test 7: Error handling - empty text"
curl -s -X POST \
  -H "Authorization: Bearer $REPLICATE_API_TOKEN" \
  -H "Content-Type: application/json" \
  -H "Prefer: wait" \
  -d '{
    "version": "VERSION_HASH_HERE",
    "input": {
      "text": ""
    }
  }' \
  https://api.replicate.com/v1/predictions

