// script.js

document.addEventListener("DOMContentLoaded", function() {
    const form = document.getElementById('prediction-form');
    const loadingSpinner = document.getElementById('loading-spinner');
    
    form.addEventListener('submit', function(event) {
        // Show the loading spinner when the form is submitted
        loadingSpinner.style.display = 'block';
        
        // Optionally, disable the submit button to prevent multiple submissions
        const submitButton = form.querySelector('input[type="submit"]');
        submitButton.disabled = true;

        // Allow the form to be submitted after the loading spinner is displayed
        setTimeout(() => {
            // Normally, you'd want to handle the form submission to the backend here (e.g., via an AJAX call)
            // However, since the form is set to submit traditionally, we'll just let it submit after the spinner shows up
            form.submit(); 
        }, 500);  // Optional delay to show spinner for a brief moment before submission
    });
});
