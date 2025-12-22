// Block new tabs
chrome.tabs.onCreated.addListener(function (tab) {
    chrome.tabs.query({}, function (tabs) {
        // Allow only one tab (the exam tab)
        if (tabs.length > 1) {
            chrome.tabs.remove(tab.id);
            console.log("Blocked new tab creation");
        }
    });
});

// Block new windows
chrome.windows.onCreated.addListener(function (window) {
    chrome.tabs.query({}, function (tabs) {
        if (tabs.length > 0) {
            chrome.windows.remove(window.id);
            console.log("Blocked new window creation");
        }
    });
});

// Heartbeat to server
chrome.alarms.create("heartbeat", { periodInMinutes: 0.5 });

chrome.alarms.onAlarm.addListener((alarm) => {
    if (alarm.name === "heartbeat") {
        fetch('http://127.0.0.1:5000/api/lockdown_heartbeat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ status: 'active', timestamp: Date.now() })
        }).catch(err => console.log("Heartbeat failed", err));
    }
});
