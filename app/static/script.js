document.getElementById('searchForm').addEventListener('submit', function (event) {
    event.preventDefault();

    // Get form input
    const search = document.getElementById('search').value.trim();
    let isValid = true;

    // Clear previous error messages
    document.getElementById('searchTextError').textContent = '';

    // Validate search text input
    if (!search || search.length === 0) {
        document.getElementById('searchTextError').textContent = 'Please enter a valid search text.';
        isValid = false;
    }

    if (isValid) {
        predictRelevantContent(search);
    }
});

function predictRelevantContent(search) {
    const apiUrl = `/predict?search=${encodeURIComponent(search)}`;

    fetch(apiUrl)
        .then(response => response.json())
        .then(data => {
            if (Array.isArray(data) && data.length > 0) {
                const resultContainer = document.getElementById('resultContainer');
                const searchResultElement = document.getElementById('searchResult');

                searchResultElement.innerHTML = '';

                data.forEach(word => {
                    searchResultElement.innerHTML += `${word}`;
                });
                
                resultContainer.style.display = 'block';
            } else {
                console.error('Unexpected API response format:', data);
            }
        }).catch(error => console.error('Error:', error));
}