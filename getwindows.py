import pygetwindow as gw
import time

# Store the last active window and a list to store all switches
last_window = None
switch_count = 0
window_history = []  # This will store the sequence of windows/tabs switched to
started_tracking = False  # Flag to start counting after Visual Studio Code

# List of restricted apps or processes (you can add more apps here)
restricted_apps = [
    "Google Chrome", "Brave", "Firefox",  # Browsers
    "YouTube", "VLC Media Player", "Media Player",  # Video players
    "Valorant", "GTA5", "Steam",  # Games
    "Discord", "Skype",  # Communication apps
]

# Set to store any restricted apps that are opened
opened_restricted_apps = set()

def get_active_window():
    # Get the title of the currently active window
    window = gw.getActiveWindow()
    if window:
        return window.title
    return None

def track_window_switches():
    global last_window, switch_count, window_history, started_tracking, opened_restricted_apps

    try:
        while True:
            current_window = get_active_window()

            if current_window != last_window:
                # Skip the first window (VS Code)
                if not started_tracking and "Visual Studio Code" in current_window:
                    last_window = current_window
                    print("Started tracking after Visual Studio Code.")
                    continue  # Do not count this as a switch
                
                if not started_tracking:
                    # Begin tracking when the first switch occurs after VS Code
                    started_tracking = True
                    print(f"Started tracking after: {current_window}")

                if current_window is not None:
                    print(f"Switched to window: {current_window}")
                    switch_count += 1
                    window_history.append(current_window)  # Store the switched window

                    # Check if the current window matches any restricted apps
                    for app in restricted_apps:
                        if app in current_window:
                            opened_restricted_apps.add(app)

                last_window = current_window

            # Display the number of switches made so far
            print(f"Switch count: {switch_count}", end='\r')

            # Check every 0.5 seconds
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nProgram terminated.")
        display_window_history()

def display_window_history():
    global window_history, opened_restricted_apps
    print("\nWindow Switch History:")
    if window_history:
        for idx, window in enumerate(window_history, 1):
            print(f"{idx}. {window}")
    else:
        print("No windows were switched.")

    # Check and display any restricted apps that were opened
    if opened_restricted_apps:
        print("\nWarning: The following restricted apps were opened during the session:")
        for app in opened_restricted_apps:
            print(f"- {app}")
    else:
        print("\nNo restricted apps were opened during the session.")

if __name__ == "__main__":
    track_window_switches()
