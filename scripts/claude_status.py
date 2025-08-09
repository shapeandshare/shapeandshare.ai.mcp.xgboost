#!/usr/bin/env python3
"""Check Claude Desktop configuration status."""

import json
import os
import sys

def main():
    claude_config_dir = os.path.expanduser("~/Library/Application Support/Claude")
    claude_config_file = os.path.join(claude_config_dir, "claude_desktop_config.json")
    
    if not os.path.exists(claude_config_file):
        print(f"❌ Claude Desktop config not found at: {claude_config_file}")
        sys.exit(1)
    
    try:
        with open(claude_config_file, 'r') as f:
            config = json.load(f)
        
        servers = config.get('mcpServers', {})
        
        if 'xgboost' in servers:
            print('✅ MCP XGBoost server is configured in Claude Desktop')
            xgb_config = servers['xgboost']
            print(f'   Command: {xgb_config.get("command", "N/A")}')
            print(f'   Args: {" ".join(xgb_config.get("args", []))}')
        else:
            print('❌ MCP XGBoost server is NOT configured in Claude Desktop')
        
        print(f'Total MCP servers configured: {len(servers)}')
        for name in servers.keys():
            print(f'  - {name}')
            
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing Claude Desktop config: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error reading Claude Desktop config: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
