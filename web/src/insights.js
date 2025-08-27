// ------------- insights.js (Robust versjon) -------------

document.addEventListener('DOMContentLoaded', () => {
    // Referanser til HTML-elementer
    const propertyTypeFilter = document.getElementById('property-type-filter');
    const aggregationLevelFilter = document.getElementById('aggregation-level-filter');
    const filterValueInput = document.getElementById('filter-value');
    const applyFiltersBtn = document.getElementById('apply-filters-btn');
    const rawDataTable = document.getElementById('raw-data-table');
    const loadingSpinner = document.getElementById('loading-spinner');

    const API_BASE_URL = 'https://api-86613370495.europe-west1.run.app'; // Sjekk at denne er riktig

    const showLoading = (isLoading) => {
        loadingSpinner.classList.toggle('hidden', !isLoading);
    };

    async function fetchData(path, params = {}) {
        const url = new URL(`${API_BASE_URL}${path}`);
        Object.entries(params).forEach(([key, value]) => {
            if (value) url.searchParams.append(key, value);
        });

        console.log(`Henter data fra: ${url}`); // DEBUG: Viser hvilken URL som kalles

        try {
            const response = await fetch(url);
            const data = await response.json();

            if (!response.ok) {
                // Hvis serveren svarer med en feil (f.eks. 400, 500)
                throw new Error(`API-feil (${response.status}): ${data.detail || 'Ukjent feil'}`);
            }

            console.log(`Mottatt data for ${path}:`, data); // DEBUG: Viser rådataen i konsollen
            return data;

        } catch (error) {
            console.error(`Feil under fetch-kall til ${url}:`, error);
            alert(`En feil oppstod under lasting av data. Sjekk konsollen for detaljer.\nFeil: ${error.message}`);
            return null; // Returner null ved feil
        }
    }

    function drawCharts(aggregatedData, aggregationLevel, propertyType) {
        // Tømmer gamle grafer
        Plotly.purge('price-chart');
        Plotly.purge('count-chart');

        if (!aggregatedData || !Array.isArray(aggregatedData) || aggregatedData.length === 0) {
            console.log("Ingen aggregerte data å vise i grafer.");
            return;
        }

        try {
            const keyField = aggregationLevel === 'by-district-oslo' ? 'district_name' : aggregationLevel;
            const labels = aggregatedData.map(d => d[keyField]);

            const priceTrace = {
                x: labels,
                y: aggregatedData.map(d => d.price_pr_sqm),
                type: 'bar',
            };
            const priceLayout = {
                title: `Gj.snittlig Kvm-pris (${propertyType})`,
                xaxis: { title: aggregationLevel },
                yaxis: { title: 'Pris (NOK)' }
            };
            Plotly.newPlot('price-chart', [priceTrace], priceLayout, {responsive: true});

            const countTrace = {
                labels: labels,
                values: aggregatedData.map(d => d.n),
                type: 'pie',
                hole: .4
            };
            const countLayout = { title: `Antall Boliger` };
            Plotly.newPlot('count-chart', [countTrace], countLayout, {responsive: true});
        } catch (error) {
            console.error("Feil under tegning av grafer:", error);
            alert("Kunne ikke tegne grafer. Sjekk at dataformatet er korrekt.");
        }
    }

    function populateRawDataTable(rawData) {
        const thead = rawDataTable.querySelector('thead');
        const tbody = rawDataTable.querySelector('tbody');
        thead.innerHTML = '';
        tbody.innerHTML = '';

        if (!rawData || !Array.isArray(rawData) || rawData.length === 0) {
            tbody.innerHTML = '<tr><td colspan="5">Ingen kildedata funnet. Spesifiser et gyldig filter for å se data.</td></tr>';
            return;
        }

        try {
            const headers = ['adresse', 'price', 'price_pr_sqm', 'salgstid', 'url'];
            const headerRow = document.createElement('tr');
            headers.forEach(h => {
                const th = document.createElement('th');
                th.textContent = h;
                headerRow.appendChild(th);
            });
            thead.appendChild(headerRow);

            rawData.forEach(item => {
                const row = document.createElement('tr');
                headers.forEach(header => {
                    const td = document.createElement('td');
                    if (header === 'url' && item[header]) {
                        const a = document.createElement('a');
                        a.href = item[header];
                        a.textContent = 'Link til annonse';
                        a.target = '_blank';
                        td.appendChild(a);
                    } else {
                        // Viser 'N/A' hvis feltet er null eller undefined
                        td.textContent = item[header] != null ? item[header] : 'N/A';
                    }
                    row.appendChild(td);
                });
                tbody.appendChild(row);
            });
        } catch (error) {
            console.error("Feil ved bygging av rådatatabell:", error);
            alert("Kunne ikke bygge rådatatabellen. Sjekk dataformat.");
        }
    }

    async function updateDashboard() {
        showLoading(true);

        const propertyType = propertyTypeFilter.value;
        const aggregationLevel = aggregationLevelFilter.value;
        const filterValue = filterValueInput.value.trim();

        // Hent aggregerte data
        let aggregatedData;
        if (aggregationLevel === 'by-district-oslo') {
            aggregatedData = await fetchData(`/homes/by-district-oslo`, { name: filterValue });
        } else {
            aggregatedData = await fetchData(`/homes/${propertyType}/${aggregationLevel}`, { filter_value: filterValue });
        }
        drawCharts(aggregatedData, aggregationLevel, propertyType);

        // Hent rådata KUN hvis et filter er satt
        if (filterValue && (aggregationLevel === 'municipality' || aggregationLevel === 'postal_code')) {
            const rawDataParams = { property_type: propertyType };
            rawDataParams[aggregationLevel] = filterValue; // Dynamisk nøkkel

            const rawData = await fetchData('/homes', rawDataParams);
            populateRawDataTable(rawData);
        } else {
            populateRawDataTable([]); // Tøm tabellen hvis ikke noe filter er satt
        }

        showLoading(false);
    }

    applyFiltersBtn.addEventListener('click', updateDashboard);
    updateDashboard(); // Kjør en gang ved oppstart
});