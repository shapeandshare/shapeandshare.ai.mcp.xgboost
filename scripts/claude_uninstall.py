#!/usr/bin/env python3
"""Remove MCP XGBoost server configuration from Claude Desktop."""

import json
import os
import sys

def main():
    claude_config_dir = os.path.expanduser("~/Library/Application Support/Claude")
    claude_config_file = os.path.join(claude_config_dir, "claude_desktop_config.json")
    
    if not os.path.exists(claude_config_file):
        print("ℹ️  No Claude Desktop config found. Nothing to remove.")
        return
    
    try:
        with open(claude_config_file, 'r') as f:
            config = json.load(f)
        
        servers = config.get('mcpServers', {})
        
        if 'xgboost' in servers:
            del servers['xgboost']
            with open(claude_config_file, 'w') as f:
                json.dump(config, f, indent=2)
            print('✅ MCP XGBoost server removed from Claude Desktop config.')
        else:
            print('ℹ️  MCP XGBoost server not found in Claude Desktop config.')
            
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing Claude Desktop config: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error updating Claude Desktop config: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
