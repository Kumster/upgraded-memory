import os
from openai import OpenAI
from dotenv import load_dotenv
import time

# Load API key
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

print(f"✓ API Key: {api_key[:20]}...")
print(f"✓ Full Key Length: {len(api_key)} characters")

client = OpenAI(api_key=api_key)

print("\n" + "="*60)
print("TEST 1: Get Account/Organization Info")
print("="*60)
try:
    # Try to get organization info
    response = client.models.retrieve("gpt-3.5-turbo")
    print(f"✅ Can access model info for gpt-3.5-turbo")
    print(f"   Model ID: {response.id}")
    print(f"   Owned by: {response.owned_by}")
except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "="*60)
print("TEST 2: Minimal Request (5 tokens)")
print("="*60)
try:
    resp = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hi"}],
        max_tokens=5
    )
    print(f"✅ SUCCESS!")
    print(f"   Response: {resp.choices[0].message.content}")
    print(f"   Tokens used: {resp.usage.total_tokens}")
except Exception as e:
    error_str = str(e)
    print(f"❌ ERROR: {error_str}")
    
    # Detailed error analysis
    if "429" in error_str:
        print("\n🔍 Error 429 Analysis:")
        if "insufficient_quota" in error_str.lower():
            print("   → Type: INSUFFICIENT_QUOTA")
            print("   → This means: No credits available")
        elif "rate_limit" in error_str.lower():
            print("   → Type: RATE_LIMIT_EXCEEDED")
            print("   → This means: Too many requests")
        else:
            print("   → Type: Unknown 429 error")
    
    print("\n📋 Full error message:")
    print(error_str)

print("\n" + "="*60)
print("TEST 3: Wait 10 seconds and retry")
print("="*60)
print("Waiting 10 seconds...")
time.sleep(10)

try:
    resp = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Test"}],
        max_tokens=5
    )
    print(f"✅ SUCCESS after waiting!")
    print(f"   Response: {resp.choices[0].message.content}")
except Exception as e:
    print(f"❌ Still failing: {str(e)}")

print("\n" + "="*60)
print("TEST 4: Try Different Model (gpt-3.5-turbo-0125)")
print("="*60)
try:
    resp = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[{"role": "user", "content": "Hi"}],
        max_tokens=5
    )
    print(f"✅ SUCCESS with specific version!")
    print(f"   Response: {resp.choices[0].message.content}")
except Exception as e:
    print(f"❌ ERROR: {str(e)}")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print("If Test 1 passed but Test 2 failed:")
print("  → The key is valid but has restrictions")
print("\nIf all tests fail with 429:")
print("  → Quota issue OR rate limiting")
print("\nIf tests pass after waiting:")
print("  → Rate limiting (not quota)")
print("\n💡 Ask your professor:")
print("  1. Is this key restricted to certain models?")
print("  2. Does it have per-minute/per-hour limits?")
print("  3. Is it restricted to certain IP addresses?")
print("  4. Does it require being on GWU network/VPN?")