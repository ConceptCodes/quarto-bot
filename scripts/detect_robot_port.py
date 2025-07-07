#!/usr/bin/env python3
"""
SO-ARM101 Serial Port Detection Script
Python version of the port detection utility.
"""

import serial.tools.list_ports
import os
import time
from typing import List


def list_ports() -> List[str]:
    """List all available serial ports."""
    ports = serial.tools.list_ports.comports()
    return [port.device for port in ports]


def prompt(msg: str) -> None:
    """Display a message and wait for user input."""
    input(msg)


def save_to_env(port: str) -> None:
    """Save the detected port to .env file."""
    env_file = ".env"
    env_content = ""
    
    # Read existing .env file if it exists
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            env_content = f.read()
    
    # Remove any existing ROBOT_PORT line
    lines = env_content.split('\n')
    filtered_lines = [line for line in lines if not line.startswith('ROBOT_PORT=')]
    
    # Add the new ROBOT_PORT line
    filtered_lines.append(f'ROBOT_PORT={port}')
    
    # Write back to .env file
    new_env_content = '\n'.join(filtered_lines).strip() + '\n'
    with open(env_file, 'w') as f:
        f.write(new_env_content)


def main():
    """Main function for port detection."""
    print("\n=== SO-ARM101 Serial Port Detection ===\n")
    
    print("1. Please plug in your robot and press Enter.")
    prompt("")
    
    print("Detecting ports...")
    before = list_ports()
    print(f"Detected ports: {before}")
    
    print("\n2. Now unplug your robot and press Enter.")
    prompt("")
    
    print("Detecting ports...")
    after = list_ports()
    print(f"Detected ports: {after}")
    
    # Find the difference (ports that were removed)
    diff = [port for port in before if port not in after]
    
    if len(diff) == 1:
        detected_port = diff[0]
        print(f"\n✅ Detected robot port: {detected_port}")
        
        # Save to .env
        try:
            save_to_env(detected_port)
            print("Saved to .env.")
        except Exception as e:
            print(f"Warning: Could not save to .env file: {e}")
            print(f"Please manually set ROBOT_PORT={detected_port} in your .env file")
            
    elif len(diff) > 1:
        print("\n⚠️  Multiple ports removed. Please try again and ensure only the robot is unplugged.")
        print(f"Removed ports: {diff}")
    else:
        print("\n❌ Could not detect a unique port. Please try again.")
        print("Make sure the robot was properly connected and then disconnected.")


def interactive_port_selection():
    """Interactive port selection if automatic detection fails."""
    print("\n=== Manual Port Selection ===")
    ports = list_ports()
    
    if not ports:
        print("No serial ports detected.")
        return
    
    print("Available serial ports:")
    for i, port in enumerate(ports, 1):
        try:
            # Try to get more info about the port
            port_info = next(serial.tools.list_ports.comports())
            for p in serial.tools.list_ports.comports():
                if p.device == port:
                    port_info = p
                    break
            
            description = getattr(port_info, 'description', 'Unknown')
            manufacturer = getattr(port_info, 'manufacturer', 'Unknown')
            print(f"  {i}. {port} - {description} ({manufacturer})")
        except:
            print(f"  {i}. {port}")
    
    try:
        choice = input(f"\nSelect port (1-{len(ports)}) or 'q' to quit: ").strip()
        
        if choice.lower() == 'q':
            return
        
        port_index = int(choice) - 1
        if 0 <= port_index < len(ports):
            selected_port = ports[port_index]
            print(f"Selected port: {selected_port}")
            
            # Ask for confirmation
            confirm = input("Save this port to .env? (y/n): ").strip().lower()
            if confirm == 'y':
                save_to_env(selected_port)
                print("Port saved to .env file!")
        else:
            print("Invalid selection.")
            
    except ValueError:
        print("Invalid input.")


def test_port(port: str):
    """Test if a port can be opened."""
    try:
        with serial.Serial(port, 9600, timeout=1) as ser:
            print(f"✅ Port {port} is accessible")
            return True
    except Exception as e:
        print(f"❌ Cannot access port {port}: {e}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="SO-ARM101 Serial Port Detection")
    parser.add_argument("--manual", action="store_true", 
                       help="Manual port selection mode")
    parser.add_argument("--test", type=str, metavar="PORT",
                       help="Test if a specific port is accessible")
    parser.add_argument("--list", action="store_true",
                       help="List all available ports")
    
    args = parser.parse_args()
    
    if args.test:
        test_port(args.test)
    elif args.list:
        ports = list_ports()
        print("Available serial ports:")
        for port in ports:
            print(f"  {port}")
    elif args.manual:
        interactive_port_selection()
    else:
        main()
