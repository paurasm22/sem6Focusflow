import pygetwindow as gw
import time
# blink,gaze,sleep 
# Store the last active window and a list to store all switches
last_window = None
window_history = []  # This will store the sequence of windows/tabs switched to
started_tracking = False  # Flag to start counting after Visual Studio Code

# List of restricted apps or processes (you can add more apps here)
restricted_apps = [
    "Google Chrome", "Brave", "Firefox",  # Browsers
    "YouTube", "VLC Media Player", "Media Player",  # Video players
    "Valorant", "GTA5", "Steam",  # Games
    "Discord", "Skype", "Chatgpt"  # Communication apps
]

# Set to store any restricted apps that are opened
opened_restricted_apps = set()

# Dictionary to store the time each window is open (in seconds)
window_times = {}

# Track the start time of the session
session_start_time = time.time()
session_duration = 60  # 60 seconds for session tracking

def get_active_window():
    # Get the title of the currently active window
    window = gw.getActiveWindow()
    if window:
        return window.title
    return None

def track_window_switches():
    global last_window, window_history, started_tracking, opened_restricted_apps, window_times

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
                    # Record the time spent on the last window, but only after the first window switch
                    if last_window and window_history:
                        time_spent = time.time() - window_history[-1][1]
                        if last_window in window_times:
                            window_times[last_window] += time_spent
                        else:
                            window_times[last_window] = time_spent

                    # Store the current window and timestamp
                    window_history.append((current_window, time.time()))
                    print(f"Switched to window: {current_window}")

                    # Check if the current window matches any restricted apps (case-insensitive match)
                    for app in restricted_apps:
                        if app.lower() in current_window.lower():  # Case-insensitive match
                            opened_restricted_apps.add(app)

                last_window = current_window

            # Calculate session elapsed time
            session_elapsed_time = time.time() - session_start_time

            # If the session time exceeds 1 minute, break out of the loop
            if session_elapsed_time >= session_duration:
                # Record the last window's time
                if last_window and window_history:
                    time_spent = time.time() - window_history[-1][1]
                    if last_window in window_times:
                        window_times[last_window] += time_spent
                    else:
                        window_times[last_window] = time_spent

                # Calculate and display time percentages
                total_time = sum(window_times.values())
                print("\nTime Spent on Each Window:")
                for window, time_spent in window_times.items():
                    print(f"{window}: {time_spent:.2f} seconds")

                # Calculate the percentage of time spent away from VS Code
                vs_code_opened = False
                for window in window_times:
                    if "Visual Studio Code" in window:
                        vs_code_opened = True
                        vs_code_time = window_times[window]
                        break

                if vs_code_opened:
                    away_time = total_time - vs_code_time
                    away_percentage = (away_time / total_time) * 100 if total_time > 0 else 0
                    print(f"\nPercentage of time away from Visual Studio Code: {away_percentage:.2f}%")
                else:
                    print("\nVisual Studio Code was not opened during the session.")

                # Break the loop after 1 minute
                break

            # Display the switch count and window information in real time
            print(f"Switch count: {len(window_history)}", end='\r')

            # Check every 0.5 seconds
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\nProgram terminated.")
        display_window_history()

def display_window_history():
    global window_history, opened_restricted_apps
    print("\nWindow Switch History:")
    if window_history:
        for idx, (window, _) in enumerate(window_history, 1):
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
