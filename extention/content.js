// This is like a while True loop, but it runs every 3 seconds
setInterval(function() {

    // 1. Find all comments (Like finding all rows in a DataFrame)
    let comments = document.querySelectorAll("#content-text");

    comments.forEach(comment => {
        // 2. Only process comments we haven't seen before
        if (!comment.hasAttribute('data-checked')) {
            
            let textValue = comment.innerText;

            // 3. SEND DATA TO YOUR PYTHON BRAIN (The "Fetch" is like a Request)
            fetch('http://127.0.0.1:5000/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ "text": textValue }) // Sending text to Flask
            })
            .then(response => response.json()) // Waiting for Python's answer
            .then(data => {
                // 4. If Python said "spam", hide the comment
                if (data.prediction === "spam") {
                    comment.innerText = "🚨 [AI BLOCKED THIS SPAM]";
                    comment.style.color = "red";
                    comment.style.backgroundColor = "#ffebee";
                }
            });

            // Mark as checked so we don't ask Python about the same comment twice
            comment.setAttribute('data-checked', 'true');
        }
    });

}, 3000);