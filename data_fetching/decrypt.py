import base64
import zlib
import json
import re

def decode_coinglass_heatmap(raw_encoded_string):
    """
    Decodes CoinGlass API data.
    Handles Base64 cleaning, string reversal, and Zlib/Pako decompression.
    """
    try:
        # 1. CLEANING: Remove whitespace and trailing metadata (e.g., " 57.1 kb")
        # This regex removes any character not in the Base64 alphabet
        clean_data = re.sub(r'[^A-Za-z0-9+/=]', '', raw_encoded_string)
        
        # 2. DECODING STRATEGY A: Standard Base64 -> Zlib (Most likely)
        try:
            binary_data = base64.b64decode(clean_data)
            # wbits=zlib.MAX_WBITS | 32 automatically detects Gzip or Zlib headers
            decompressed = zlib.decompress(binary_data, zlib.MAX_WBITS | 32)
            return json.loads(decompressed.decode('utf-8'))
        except Exception:
            pass

        # 3. DECODING STRATEGY B: String Reverse -> Base64 -> Zlib
        # Some versions of the Coinglass frontend reverse the string before sending
        try:
            binary_data = base64.b64decode(clean_data[::-1])
            decompressed = zlib.decompress(binary_data, zlib.MAX_WBITS | 32)
            return json.loads(decompressed.decode('utf-8'))
        except Exception:
            pass

        # 4. DECODING STRATEGY C: Raw Inflate (No Headers)
        # Equivalent to pako.inflate(binary, { raw: true })
        try:
            binary_data = base64.b64decode(clean_data)
            decompressed = zlib.decompress(binary_data, -zlib.MAX_WBITS)
            return json.loads(decompressed.decode('utf-8'))
        except Exception:
            pass

        return {"error": "All common decoding methods failed. The site may be using a custom alphabet key."}

    except Exception as e:
        return {"error": f"Unexpected error during execution: {str(e)}"}

# ==========================================
# EXAMPLE USAGE
# ==========================================
if __name__ == "__main__":
    # Replace this with the actual long string from your Network 'data' field
    sample_data = "eJztmM9v2jAUhV/8KXr1n3u+"
    
    result = decode_coinglass_heatmap(sample_data)
    
    if "error" in result:
        print(result["error"])
    else:
        # Print the first few keys to verify (e.g., 'price', 'time', 'list')
        print("Keys found in decoded data:", result.keys())
        # To save to a file:
        # with open('heatmap_data.json', 'w') as f:
        #     json.dump(result, f)