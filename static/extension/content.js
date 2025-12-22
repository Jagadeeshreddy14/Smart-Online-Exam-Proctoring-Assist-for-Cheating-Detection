// Disable context menu
document.addEventListener('contextmenu', event => {
    event.preventDefault();
    return false;
}, true);

// Disable keyboard shortcuts
document.addEventListener('keydown', function (e) {
    // Block: Ctrl+C, Ctrl+V, Ctrl+X, Ctrl+A, Ctrl+P, Ctrl+S, Ctrl+U (Source), Ctrl+Shift+I (DevTools)
    // Mac uses Meta key (Command), Windows uses Ctrl
    const isControl = e.ctrlKey || e.metaKey;

    if (isControl) {
        const key = e.key.toLowerCase();
        if (['c', 'v', 'x', 'a', 'p', 's', 'u'].includes(key)) {
            e.preventDefault();
            e.stopPropagation();
            // alert("Restricted Action: Clipboard and shortcuts are disabled.");
            return false;
        }
    }

    // Block DevTools shortcuts separately if needed (Ctrl+Shift+I/J)
    if (isControl && e.shiftKey && (e.key.toLowerCase() === 'i' || e.key.toLowerCase() === 'j')) {
        e.preventDefault();
        return false;
    }

    // Print Screen
    if (e.key === 'PrintScreen') {
        e.preventDefault();
        // alert("Restricted Action: Screenshots are disabled.");
        return false;
    }

    // F11 / F12 prevention (Fullscreen / DevTools)
    if (e.key === 'F12' || e.key === 'F11') {
        e.preventDefault();
        return false;
    }

    // Alt+Tab (Difficult to block in JS, but we can try to consume Alt)
    if (e.altKey && e.key === 'Tab') {
        e.preventDefault();
    }
}, true);

// Selection clearing (prevent highlighting text)
document.addEventListener('selectstart', function (e) {
    e.preventDefault();
    return false;
});
