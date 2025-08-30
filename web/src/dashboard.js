const valuationButton = document.getElementById("valuation-assistant")
valuationButton.addEventListener('click', () => { window.location.href = 'index.html'; });

document.addEventListener('DOMContentLoaded', () => {
    // API Configuration
    //const API_BASE_URL = 'https://api-86613370495.europe-west1.run.app';
    const API_BASE_URL ='http://0.0.0.0:8080'

    // Columns to display
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
    const monthBtn = document.getElementById('period-month');
    const quarterBtn = document.getElementById('period-quarter');

    // State
    let currentPage = 1;
    let currentPeriod = 'quarter'; // 'month' or 'quarter'
    const limit = 50;
    let currentFilters = { url: 'https' }; // DEFAULT FILTER
    let isFetching = false;

    // Debounce function
    const debounce = (func, wait) => {
        let timeout;
        return (...args) => {
            clearTimeout(timeout);
            timeout = setTimeout(() => func.apply(this, args), wait);
        };
    };

    /**
     * Main function to fetch all dashboard data from the new endpoint.
     */
    async function fetchDashboardData() {
        if (isFetching) return;
        isFetching = true;
        loadingSpinner.classList.remove('hidden');
        tableBody.innerHTML = '';

        const params = new URLSearchParams({
            page: currentPage,
            limit: limit,
            period: currentPeriod,
            ...currentFilters
        });

        try {
            const response = await fetch(`${API_BASE_URL}/homes/dashboard-stats?${params.toString()}`);
            console.log(`${API_BASE_URL}/homes/dashboard-stats?${params.toString()}`);
            if (!response.ok) throw new Error(`API Error: ${response.statusText}`);

            const data = await response.json();

            updateKpis(data.kpis);
            populateTable(data.homes);
            updatePagination(data.homes.length);

        } catch (error) {
            console.error("Failed to fetch dashboard data:", error);
            tableBody.innerHTML = `<tr><td colspan="${COLUMNS.length}">Kunne ikke laste data.</td></tr>`;
        } finally {
            isFetching = false;
            loadingSpinner.classList.add('hidden');
        }
    }

    /**
     * Updates the KPI cards with new data.
     * @param {Object} kpis - The KPI object from the API.
     */
    function updateKpis(kpis = {}) {
        const { current, previous } = kpis;

        const calculateChange = (currentVal, previousVal) => {
            if (previousVal === 0 || previousVal == null || currentVal == null) return null;
            return ((currentVal - previousVal) / previousVal) * 100;
        };

        const formatValue = (key, value) => {
            if (value == null) return '-';
            if (key === 'avg_price') {
                return new Intl.NumberFormat('nb-NO', { style: 'currency', currency: 'NOK', maximumFractionDigits: 0 }).format(value);
            }
            return new Intl.NumberFormat('nb-NO').format(Math.round(value));
        };

        const updateCard = (key) => {
            const valueEl = document.getElementById(`kpi-${key}`);
            const changeEl = document.getElementById(`kpi-${key}-change`);

            const currentValue = current ? current[key] : null;
            const previousValue = previous ? previous[key] : null;

            valueEl.textContent = formatValue(key, currentValue);

            const change = calculateChange(currentValue, previousValue);

            if (change != null) {
                const sign = change > 0 ? '▲' : '▼';
                changeEl.textContent = `${sign} ${change.toFixed(1)}% vs forrige periode`;
                changeEl.className = `kpi-change ${change > 0 ? 'positive' : 'negative'}`;
            } else {
                changeEl.textContent = '';
            }
        };

        ['avg_price', "avg_sqm_price",'avg_sqm', 'avg_build_year', 'n_samples'].forEach(updateCard);
    }

    /**
     * Populates the main data table.
     * @param {Array<Object>} homes - Array of home objects.
     */
    function populateTable(homes) {
        if (homes.length === 0) {
            tableBody.innerHTML = `<tr><td colspan="${COLUMNS.length}">Ingen resultater funnet.</td></tr>`;
            return;
        }
        const rows = homes.map(home => {
             const tr = document.createElement('tr');
             COLUMNS.forEach(col => {
                 const td = document.createElement('td');
                 let value = home[col];
                 if (value === null || typeof value === 'undefined') value = 'N/A';
                 else if (col === 'price' || col === 'price_pr_sqm') value = Math.round(value).toLocaleString('nb-NO');
                 else if (col === 'url') {
                    td.innerHTML = `<a href="${value}" target="_blank">Link</a>`;
                    return tr.appendChild(td);
                 }
                 td.textContent = value;
                 tr.appendChild(td);
             });
             return tr;
        });
        tableBody.append(...rows);
    }

    /**
     * Initializes the table header with titles and filter inputs.
     */
    function initializeHeader() {
        const titleRow = document.createElement('tr');
        const filterRow = document.createElement('tr');

        COLUMNS.forEach(col => {
            const thTitle = document.createElement('th');
            thTitle.textContent = col.replace(/_/g, ' ').toUpperCase();
            titleRow.appendChild(thTitle);

            const thFilter = document.createElement('th');
            const input = document.createElement('input');
            input.type = 'text';
            input.placeholder = `Filtrer...`;
            input.className = 'filter-input';
            input.dataset.column = col;

            // Set default filter value from state
            if (currentFilters[col]) {
                input.value = currentFilters[col];
            }

            input.addEventListener('input', debouncedFilterChange);
            thFilter.appendChild(input);
            filterRow.appendChild(thFilter);
        });

        tableHead.append(titleRow, filterRow);
    }

    const debouncedFilterChange = debounce((event) => {
        const { column } = event.target.dataset;
        const { value } = event.target;
        if (value) {
            currentFilters[column] = value;
        } else {
            delete currentFilters[column];
        }
        currentPage = 1;
        fetchDashboardData();
    }, 500);

    function updatePagination(fetchedCount) {
        pageInfo.textContent = `Side ${currentPage}`;
        prevPageBtn.disabled = currentPage === 1;
        nextPageBtn.disabled = fetchedCount < limit;
    }

    // Event Listeners
    prevPageBtn.addEventListener('click', () => {
        if (currentPage > 1) {
            currentPage--;
            fetchDashboardData();
        }
    });

    nextPageBtn.addEventListener('click', () => {
        currentPage++;
        fetchDashboardData();
    });

    monthBtn.addEventListener('click', () => {
        currentPeriod = 'month';
        monthBtn.classList.add('active');
        quarterBtn.classList.remove('active');
        fetchDashboardData();
    });

    quarterBtn.addEventListener('click', () => {
        currentPeriod = 'quarter';
        quarterBtn.classList.add('active');
        monthBtn.classList.remove('active');
        fetchDashboardData();
    });

    // Initial Load
    initializeHeader();
    fetchDashboardData();
});