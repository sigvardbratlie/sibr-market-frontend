document.addEventListener('DOMContentLoaded', () => {
    // API Configuration
    const API_BASE_URL = 'https://api-86613370495.europe-west1.run.app';

    // Columns to display, in order
    const COLUMNS = [
        "item_id", "address", "municipality", "county", "price", "price_pr_sqm",
        "property_type", "usable_area", "bedrooms", "build_year",
        "last_updated", "dealer", "url"
    ];

    // DOM Elements
    const tableHead = document.getElementById('table-head');
    const tableBody = document.getElementById('table-body');
    const loadingSpinner = document.getElementById('loading-spinner');
    const prevPageBtn = document.getElementById('prev-page');
    const nextPageBtn = document.getElementById('next-page');
    const pageInfo = document.getElementById('page-info');

    // State
    let currentPage = 1;
    const limit = 50; // 50 rows per page
    let currentFilters = {};
    let isFetching = false;

    /**
     * Creates a debounced function that delays invoking func until after wait milliseconds
     * have elapsed since the last time the debounced function was invoked.
     * @param {Function} func The function to debounce.
     * @param {number} wait The number of milliseconds to delay.
     * @returns {Function} Returns the new debounced function.
     */
    const debounce = (func, wait) => {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    };

    /**
     * Fetches home data from the API based on current filters and page.
     */
    async function fetchHomes() {
        if (isFetching) return;
        isFetching = true;
        loadingSpinner.classList.remove('hidden');
        tableBody.innerHTML = ''; // Clear table while loading

        const params = new URLSearchParams({
            page: currentPage,
            limit: limit,
            ...currentFilters
        });

        try {
            const response = await fetch(`${API_BASE_URL}/homes/search?${params.toString()}`);
            if (!response.ok) {
                throw new Error(`API Error: ${response.statusText}`);
            }
            const data = await response.json();
            populateTable(data);
            updatePagination(data.length);
        } catch (error) {
            console.error("Failed to fetch homes:", error);
            tableBody.innerHTML = `<tr><td colspan="${COLUMNS.length}">Kunne ikke laste data. Prøv igjen senere.</td></tr>`;
        } finally {
            isFetching = false;
            loadingSpinner.classList.add('hidden');
        }
    }

    /**
     * Populates the table with data from the API.
     * @param {Array<Object>} homes - An array of home objects.
     */
    function populateTable(homes) {
        if (homes.length === 0) {
            tableBody.innerHTML = `<tr><td colspan="${COLUMNS.length}">Ingen resultater funnet for dine filter.</td></tr>`;
            return;
        }

        const rows = homes.map(home => {
            const tr = document.createElement('tr');
            COLUMNS.forEach(col => {
                const td = document.createElement('td');
                let value = home[col];

                // Formatting
                if (value === null || typeof value === 'undefined') {
                    value = 'N/A';
                } else if (col === 'price' || col === 'price_pr_sqm') {
                    value = new Intl.NumberFormat('nb-NO', { style: 'currency', currency: 'NOK', minimumFractionDigits: 0 }).format(value);
                } else if (col === 'url') {
                    value = `<a href="${value}" target="_blank">Link</a>`;
                    td.innerHTML = value;
                }

                if (col !== 'url') {
                   td.textContent = value;
                }
                tr.appendChild(td);
            });
            return tr;
        });

        tableBody.append(...rows);
    }

    /**
     * Updates the pagination buttons based on the current state.
     * @param {number} fetchedCount - The number of items fetched in the current request.
     */
    function updatePagination(fetchedCount) {
        pageInfo.textContent = `Side ${currentPage}`;
        prevPageBtn.disabled = currentPage === 1;
        nextPageBtn.disabled = fetchedCount < limit; // Disable next if we got less than a full page
    }

    /**
     * Initializes the table header with titles and filter inputs.
     */
    function initializeHeader() {
        const titleRow = document.createElement('tr');
        const filterRow = document.createElement('tr');

        COLUMNS.forEach(col => {
            // Title cell
            const thTitle = document.createElement('th');
            thTitle.textContent = col.replace(/_/g, ' ').toUpperCase();
            titleRow.appendChild(thTitle);

            // Filter cell
            const thFilter = document.createElement('th');
            const input = document.createElement('input');
            input.type = 'text';
            input.placeholder = `Filtrer ${col}...`;
            input.className = 'filter-input';
            input.dataset.column = col;
            input.addEventListener('input', debouncedFilterChange);
            thFilter.appendChild(input);
            filterRow.appendChild(thFilter);
        });

        tableHead.appendChild(titleRow);
        tableHead.appendChild(filterRow);
    }

    // Debounced version of the handler
    const debouncedFilterChange = debounce((event) => {
        const { column } = event.target.dataset;
        const { value } = event.target;

        if (value) {
            currentFilters[column] = value;
        } else {
            delete currentFilters[column];
        }

        currentPage = 1; // Reset to first page on new filter
        fetchHomes();
    }, 500); // 500ms delay

    // Event Listeners for pagination
    prevPageBtn.addEventListener('click', () => {
        if (currentPage > 1) {
            currentPage--;
            fetchHomes();
        }
    });

    nextPageBtn.addEventListener('click', () => {
        currentPage++;
        fetchHomes();
    });

    // Initial load
    initializeHeader();
    fetchHomes();
});