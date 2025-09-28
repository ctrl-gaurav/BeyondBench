// Enhanced Interactive BeyondBench Leaderboard
// Global variables
let filteredData = [];
let currentSort = { column: 'rank', ascending: true };
let selectedModels = new Set();
let fabMenuOpen = false;

// Initialize particle system
function createParticles() {
    const particlesContainer = document.querySelector('.particles');
    if (!particlesContainer) return;

    const particleCount = 50;

    for (let i = 0; i < particleCount; i++) {
        const particle = document.createElement('div');
        particle.className = 'particle';

        // Random position
        particle.style.left = Math.random() * 100 + '%';

        // Random animation delay
        particle.style.animationDelay = Math.random() * 8 + 's';

        // Random animation duration
        particle.style.animationDuration = (Math.random() * 3 + 5) + 's';

        // Random opacity
        particle.style.opacity = Math.random() * 0.5 + 0.1;

        particlesContainer.appendChild(particle);
    }
}

// Generate author information
function generateAuthorInfo() {
    const authorsContainer = document.querySelector('.authors');
    const affiliationContainer = document.querySelector('.affiliations');
    const correspondenceContainer = document.querySelector('.correspondence');

    if (!authorsContainer || !affiliationContainer || !correspondenceContainer) return;

    // Generate authors
    let authorsHTML = '';
    paperInfo.authors.forEach((author, index) => {
        const superscript = author.affiliation === 'vt' ? '♥' : '♠';
        const correspondingMark = author.corresponding ? '*' : '';
        authorsHTML += `<span class="author">${author.name}<sup>${superscript}${correspondingMark}</sup></span>`;
        if (index < paperInfo.authors.length - 1) {
            authorsHTML += ', ';
        }
    });

    // Generate affiliations
    let affiliationsHTML = '';
    for (const [key, value] of Object.entries(paperInfo.affiliations)) {
        const symbol = key === 'vt' ? '♥' : '♠';
        affiliationsHTML += `<div class="affiliation"><sup>${symbol}</sup>${value}</div>`;
    }

    // Generate correspondence
    const correspondingAuthors = paperInfo.authors.filter(author => author.corresponding);
    let correspondenceHTML = '';
    if (correspondingAuthors.length > 0) {
        const emails = correspondingAuthors.map(author => author.email).join(', ');
        correspondenceHTML = `<div class="correspondence"><strong>*Correspondence:</strong> ${emails}</div>`;
    }

    authorsContainer.innerHTML = authorsHTML;
    affiliationContainer.innerHTML = affiliationsHTML;
    correspondenceContainer.innerHTML = correspondenceHTML;
}

// Generate top models showcase
function updateTopModels() {
    const topModelsGrid = document.getElementById('topModelsGrid');
    if (!topModelsGrid) return;

    const topModels = rankedModelData.slice(0, 3).sort((a, b) => a.rank - b.rank);

    topModelsGrid.innerHTML = '';
    topModels.forEach(model => {
        const card = document.createElement('div');
        card.className = `top-model-card rank-${model.rank <= 3 ? model.rank : ''}`;
        card.onclick = () => showModelDetails(model);

        card.innerHTML = `
            <div class="rank">#${model.rank}</div>
            <h3 class="model-name-center">${model.model.split('/').pop()}</h3>
            <div class="model-family">${model.model.split('/')[0]}</div>
            <div class="model-stats">
                <div class="stat">
                    <span class="stat-label">Accuracy</span>
                    <span class="stat-value">${model.accuracy.toFixed(1)}%</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Instruction</span>
                    <span class="stat-value">${model.instruction.toFixed(1)}%</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Tokens</span>
                    <span class="stat-value">${model.tokens.toFixed(0)}</span>
                </div>
            </div>
        `;

        topModelsGrid.appendChild(card);
    });
}

// Initialize filters
function initFilters() {
    const minAccuracy = document.getElementById('minAccuracy');
    const maxAccuracy = document.getElementById('maxAccuracy');
    const familyFilter = document.getElementById('familyFilter');
    const sizeFilter = document.getElementById('sizeFilter');
    const quantizationFilter = document.getElementById('quantizationFilter');
    const clearFilters = document.getElementById('clearFilters');

    // Add event listeners
    [minAccuracy, maxAccuracy, familyFilter, sizeFilter, quantizationFilter].forEach(filter => {
        if (filter) {
            filter.addEventListener('change', applyFilters);
            filter.addEventListener('input', applyFilters);
        }
    });

    if (clearFilters) {
        clearFilters.addEventListener('click', () => {
            if (minAccuracy) minAccuracy.value = '';
            if (maxAccuracy) maxAccuracy.value = '';
            if (familyFilter) familyFilter.selectedIndex = 0;
            if (sizeFilter) sizeFilter.value = '';
            if (quantizationFilter) quantizationFilter.value = '';
            applyFilters();
        });
    }

    // Search functionality
    const searchInput = document.getElementById('searchInput');
    if (searchInput) {
        searchInput.addEventListener('input', applyFilters);
    }

    // Sort functionality
    const sortSelect = document.getElementById('sortSelect');
    if (sortSelect) {
        sortSelect.addEventListener('change', (e) => {
            const [column, direction] = e.target.value.split('-');
            sortTable(column, direction !== 'desc');
        });
    }
}

// Apply filters
function applyFilters() {
    const searchTerm = document.getElementById('searchInput')?.value.toLowerCase() || '';
    const minAcc = parseFloat(document.getElementById('minAccuracy')?.value) || 0;
    const maxAcc = parseFloat(document.getElementById('maxAccuracy')?.value) || 100;
    const familyFilter = document.getElementById('familyFilter')?.value || '';
    const sizeFilter = document.getElementById('sizeFilter')?.value || '';
    const quantization = document.getElementById('quantizationFilter')?.value || '';

    filteredData = rankedModelData.filter(model => {
        // Search filter
        if (searchTerm && !model.model.toLowerCase().includes(searchTerm)) {
            return false;
        }

        // Accuracy range
        if (model.accuracy < minAcc || model.accuracy > maxAcc) {
            return false;
        }

        // Family filter
        if (familyFilter) {
            const family = model.model.split('/')[0];
            if (family !== familyFilter) {
                return false;
            }
        }

        // Size filter
        if (sizeFilter) {
            const paramSize = getParameterSize(model.params);
            if (!matchesSize(paramSize, sizeFilter)) {
                return false;
            }
        }

        // Quantization filter
        if (quantization && model.quantization !== quantization) {
            return false;
        }

        return true;
    });

    renderTable(filteredData);
    updateStats();
}

// Get parameter size category
function getParameterSize(params) {
    if (params === 'Unknown' || params === '' || !params) return 'unknown';

    // Handle MOE models
    if (params.includes('MOE') || params.includes('x')) {
        const match = params.match(/(\d+(?:\.\d+)?)(?:B|x)/);
        if (match) {
            const size = parseFloat(match[1]);
            if (!isNaN(size)) {
                if (size < 3) return 'small';
                if (size <= 20) return 'medium';
                if (size <= 100) return 'large';
                return 'xlarge';
            }
        }
    }

    const size = parseFloat(params.replace(/[^0-9.]/g, ''));
    if (isNaN(size)) return 'unknown';
    if (size < 3) return 'small';
    if (size <= 20) return 'medium';
    if (size <= 100) return 'large';
    return 'xlarge';
}

// Check if size matches filter
function matchesSize(paramSize, filter) {
    if (filter === 'small') return paramSize === 'small';
    if (filter === 'medium') return paramSize === 'medium';
    if (filter === 'large') return paramSize === 'large';
    if (filter === 'xlarge') return paramSize === 'xlarge';
    return paramSize === filter;
}

// Render table
function renderTable(data) {
    const tableBody = document.getElementById('tableBody');
    if (!tableBody) return;

    tableBody.innerHTML = '';

    data.forEach((model, index) => {
        const row = document.createElement('tr');
        row.classList.add('fade-in-up');
        row.style.animationDelay = (index * 15) + 'ms';

        row.innerHTML = `
            <td><span class="rank rank-${model.rank <= 3 ? model.rank : ''}">#${model.rank}</span></td>
            <td class="model-name" title="${model.model}">${model.model.split('/').pop()}${model.params ? ` (${model.params})` : ''}</td>
            <td>${model.easy_acc.toFixed(2)}%</td>
            <td>${model.easy_inst.toFixed(1)}%</td>
            <td>${model.easy_tokens.toFixed(0)}</td>
            <td>${model.medium_acc.toFixed(2)}%</td>
            <td>${model.medium_inst.toFixed(1)}%</td>
            <td>${model.medium_tokens.toFixed(0)}</td>
            <td>${model.hard_acc.toFixed(2)}%</td>
            <td>${model.hard_inst.toFixed(1)}%</td>
            <td>${model.hard_tokens.toFixed(0)}</td>
            <td class="overall-col"><strong>${model.accuracy.toFixed(2)}%</strong></td>
            <td class="overall-col"><strong>${model.instruction.toFixed(1)}%</strong></td>
            <td class="overall-col"><strong>${model.tokens.toFixed(0)}</strong></td>
            <td>
                <div class="actions-buttons">
                    <button class="btn-icon" onclick="showModelDetails('${model.model}')" title="View Details">
                        <i class="fas fa-info-circle"></i>
                    </button>
                    <button class="btn-icon ${selectedModels.has(model.model) ? 'selected' : ''}"
                            onclick="toggleModelSelection('${model.model}')"
                            title="${selectedModels.has(model.model) ? 'Remove from Comparison' : 'Select for Comparison'}"
                            ${selectedModels.has(model.model) ? 'style="background: var(--success-color); color: white;"' : ''}>
                        <i class="fas ${selectedModels.has(model.model) ? 'fa-check' : 'fa-plus'}"></i>
                    </button>
                </div>
            </td>
        `;
        tableBody.appendChild(row);
    });
}

// Sort table
function sortTable(column, ascending = null) {
    if (ascending === null) {
        ascending = currentSort.column === column ? !currentSort.ascending : false;
    }

    currentSort = { column, ascending };

    filteredData.sort((a, b) => {
        let aVal, bVal;

        // Map column names to actual data properties
        switch(column) {
            case 'easy_acc':
                aVal = a.easy_acc;
                bVal = b.easy_acc;
                break;
            case 'easy_inst':
                aVal = a.easy_inst;
                bVal = b.easy_inst;
                break;
            case 'easy_tokens':
                aVal = a.easy_tokens;
                bVal = b.easy_tokens;
                break;
            case 'medium_acc':
                aVal = a.medium_acc;
                bVal = b.medium_acc;
                break;
            case 'medium_inst':
                aVal = a.medium_inst;
                bVal = b.medium_inst;
                break;
            case 'medium_tokens':
                aVal = a.medium_tokens;
                bVal = b.medium_tokens;
                break;
            case 'hard_acc':
                aVal = a.hard_acc;
                bVal = b.hard_acc;
                break;
            case 'hard_inst':
                aVal = a.hard_inst;
                bVal = b.hard_inst;
                break;
            case 'hard_tokens':
                aVal = a.hard_tokens;
                bVal = b.hard_tokens;
                break;
            case 'overall_acc':
                aVal = a.accuracy;
                bVal = b.accuracy;
                break;
            case 'overall_inst':
                aVal = a.instruction;
                bVal = b.instruction;
                break;
            case 'overall_tokens':
                aVal = a.tokens;
                bVal = b.tokens;
                break;
            case 'model':
                aVal = a.model.toLowerCase();
                bVal = b.model.toLowerCase();
                break;
            case 'params':
                aVal = parseFloat(a.params.replace(/[^0-9.]/g, '')) || 0;
                bVal = parseFloat(b.params.replace(/[^0-9.]/g, '')) || 0;
                break;
            case 'rank':
                aVal = a.rank;
                bVal = b.rank;
                break;
            default:
                aVal = a[column];
                bVal = b[column];
        }

        if (typeof aVal === 'string') {
            return ascending ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
        } else {
            return ascending ? aVal - bVal : bVal - aVal;
        }
    });

    renderTable(filteredData);
    updateSortIcons(column, ascending);
}

// Update sort icons
function updateSortIcons(column, ascending) {
    document.querySelectorAll('.sortable i').forEach(icon => {
        icon.className = 'fas fa-sort';
    });

    const header = document.querySelector(`th[onclick*="${column}"] i`);
    if (header) {
        header.className = ascending ? 'fas fa-sort-up' : 'fas fa-sort-down';
    }
}

// Update stats
function updateStats() {
    const stats = document.getElementById('tableStats');
    if (stats) {
        stats.textContent = `Showing ${filteredData.length} of ${rankedModelData.length} models`;
    }
}

// Show model details
function showModelDetails(modelId) {
    const model = rankedModelData.find(m => m.model === modelId);
    if (!model) return;

    const modal = document.getElementById('modelModal');
    const content = document.getElementById('modelDetails');

    if (!modal || !content) return;

    content.innerHTML = `
        <h2>${model.model}</h2>
        <div class="model-detail-grid">
            <div class="detail-card">
                <h3>Performance Metrics</h3>
                <div class="metric-row">
                    <span>Rank</span>
                    <span class="rank-${model.rank <= 3 ? model.rank : ''}">#${model.rank}</span>
                </div>
                <div class="metric-row">
                    <span>Accuracy</span>
                    <span>${model.accuracy.toFixed(2)}%</span>
                </div>
                <div class="metric-row">
                    <span>Efficiency Score</span>
                    <span>${model.efficiency.toFixed(3)}</span>
                </div>
                <div class="metric-row">
                    <span>Instruction Following</span>
                    <span>${model.instruction.toFixed(1)}%</span>
                </div>
            </div>
            <div class="detail-card">
                <h3>Model Details</h3>
                <div class="metric-row">
                    <span>Parameters</span>
                    <span>${model.params}</span>
                </div>
                <div class="metric-row">
                    <span>Quantization</span>
                    <span>${model.quantization}</span>
                </div>
            </div>
            <div class="detail-card">
                <h3>Token Statistics</h3>
                <div class="metric-row">
                    <span>Average Tokens</span>
                    <span>${model.tokens.toFixed(0)}</span>
                </div>
                <div class="metric-row">
                    <span>Average Words</span>
                    <span>${model.words.toFixed(0)}</span>
                </div>
                <div class="metric-row">
                    <span>Average Characters</span>
                    <span>${model.chars.toFixed(0)}</span>
                </div>
            </div>
        </div>
        <div class="modal-actions">
            <button class="btn ${selectedModels.has(model.model) ? 'btn-success' : 'btn-primary'}"
                    onclick="toggleModelSelection('${model.model}')">
                <i class="fas ${selectedModels.has(model.model) ? 'fa-check' : 'fa-plus'}"></i>
                ${selectedModels.has(model.model) ? 'Remove from Comparison' : 'Add to Comparison'}
            </button>
        </div>
    `;

    modal.classList.add('active');
}

// Toggle model selection for comparison
function toggleModelSelection(modelId) {
    const isSelected = selectedModels.has(modelId);

    if (isSelected) {
        selectedModels.delete(modelId);
    } else {
        selectedModels.add(modelId);
    }

    // Update the visual state of the button
    updateModelSelectionButton(modelId, !isSelected);
    updateCompareButton();
}

// Update the visual state of model selection button
function updateModelSelectionButton(modelId, isSelected) {
    // Find the button for this model
    const buttons = document.querySelectorAll(`button[onclick="toggleModelSelection('${modelId.replace(/'/g, "\\'")}')"]`);
    buttons.forEach(button => {
        const icon = button.querySelector('i');
        if (isSelected) {
            button.classList.add('selected');
            button.title = 'Remove from Comparison';
            icon.className = 'fas fa-check';
            button.style.background = 'var(--success-color)';
            button.style.color = 'white';
        } else {
            button.classList.remove('selected');
            button.title = 'Select for Comparison';
            icon.className = 'fas fa-plus';
            button.style.background = '';
            button.style.color = '';
        }
    });
}

// Update compare button
function updateCompareButton() {
    const compareBtn = document.getElementById('compareBtn');
    if (compareBtn) {
        compareBtn.innerHTML = `
            <i class="fas fa-chart-line"></i>
            Compare Models ${selectedModels.size > 0 ? `(${selectedModels.size})` : ''}
        `;
        compareBtn.disabled = selectedModels.size === 0;
    }
}

// Initialize charts
function initCharts() {
    // Set Chart.js defaults based on current theme
    const isDark = document.documentElement.getAttribute('data-theme') !== 'light';
    Chart.defaults.color = isDark ? '#e4e4e7' : '#1f2937';
    Chart.defaults.borderColor = isDark ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)';
    Chart.defaults.backgroundColor = isDark ? 'rgba(102, 126, 234, 0.1)' : 'rgba(102, 126, 234, 0.05)';

    // Accuracy vs Efficiency Chart
    const effCtx = document.getElementById('efficiencyChart');
    if (effCtx) {
        new Chart(effCtx, {
            type: 'scatter',
            data: {
                datasets: [{
                    label: 'Models',
                    data: rankedModelData.map(m => ({
                        x: m.efficiency,
                        y: m.accuracy,
                        model: m.model,
                        rank: m.rank
                    })),
                    backgroundColor: rankedModelData.map(m =>
                        m.rank === 1 ? 'rgba(255, 215, 0, 0.8)' :
                        m.rank === 2 ? 'rgba(192, 192, 192, 0.8)' :
                        m.rank === 3 ? 'rgba(205, 127, 50, 0.8)' :
                        'rgba(102, 126, 234, 0.6)'
                    ),
                    borderColor: rankedModelData.map(m =>
                        m.rank === 1 ? 'rgba(255, 215, 0, 1)' :
                        m.rank === 2 ? 'rgba(192, 192, 192, 1)' :
                        m.rank === 3 ? 'rgba(205, 127, 50, 1)' :
                        'rgba(102, 126, 234, 1)'
                    ),
                    pointRadius: 8,
                    pointHoverRadius: 12
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const point = context.raw;
                                return `${point.model.split('/').pop()}: Efficiency ${point.x.toFixed(3)}, Accuracy ${point.y.toFixed(1)}%`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        title: { display: true, text: 'Efficiency Score' },
                        grid: { color: 'rgba(255, 255, 255, 0.1)' }
                    },
                    y: {
                        title: { display: true, text: 'Accuracy (%)' },
                        grid: { color: 'rgba(255, 255, 255, 0.1)' }
                    }
                }
            }
        });
    }

    // Difficulty Scaling Chart - Shows how models perform across difficulty levels
    const difficultyScalingCtx = document.getElementById('difficultyScalingChart');
    if (difficultyScalingCtx) {
        const topModels = rankedModelData.slice(0, 10);

        new Chart(difficultyScalingCtx, {
            type: 'line',
            data: {
                labels: ['Easy', 'Medium', 'Hard'],
                datasets: topModels.map((model, index) => ({
                    label: model.model.split('/').pop(),
                    data: [model.easy_acc, model.medium_acc, model.hard_acc],
                    borderColor: `hsl(${index * 36}, 70%, 60%)`,
                    backgroundColor: `hsla(${index * 36}, 70%, 60%, 0.1)`,
                    borderWidth: 2,
                    pointRadius: 4,
                    pointHoverRadius: 6,
                    tension: 0.3
                }))
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        display: true,
                        position: 'right',
                        labels: {
                            boxWidth: 12,
                            font: { size: 10 }
                        }
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false,
                        callbacks: {
                            label: function(context) {
                                return `${context.dataset.label}: ${context.raw.toFixed(1)}%`;
                            }
                        }
                    }
                },
                interaction: {
                    mode: 'nearest',
                    axis: 'x',
                    intersect: false
                },
                scales: {
                    x: {
                        title: { display: true, text: 'Difficulty Level' },
                        grid: { color: 'rgba(255, 255, 255, 0.1)' }
                    },
                    y: {
                        title: { display: true, text: 'Accuracy (%)' },
                        beginAtZero: true,
                        max: 100,
                        grid: { color: 'rgba(255, 255, 255, 0.1)' }
                    }
                }
            }
        });
    }

    // Parameter Count vs Performance Scatter Plot
    const parameterPerformanceCtx = document.getElementById('parameterPerformanceChart');
    if (parameterPerformanceCtx) {
        // Group models by family for color coding
        const familyData = {
            'OpenAI': { models: [], color: 'rgba(255, 215, 0, 0.8)' },
            'Google': { models: [], color: 'rgba(102, 126, 234, 0.8)' },
            'Qwen': { models: [], color: 'rgba(240, 147, 251, 0.8)' },
            'Meta': { models: [], color: 'rgba(16, 185, 129, 0.8)' },
            'Microsoft': { models: [], color: 'rgba(245, 158, 11, 0.8)' },
            'Mistral': { models: [], color: 'rgba(239, 68, 68, 0.8)' },
            'Others': { models: [], color: 'rgba(107, 114, 128, 0.8)' }
        };

        // Categorize models by family
        rankedModelData.forEach(model => {
            const paramCount = parseFloat(model.params);
            const dataPoint = {
                x: paramCount,
                y: model.accuracy,
                model: model.model
            };

            if (model.model.includes('OpenAI')) familyData.OpenAI.models.push(dataPoint);
            else if (model.model.includes('Google')) familyData.Google.models.push(dataPoint);
            else if (model.model.includes('Qwen')) familyData.Qwen.models.push(dataPoint);
            else if (model.model.includes('Meta')) familyData.Meta.models.push(dataPoint);
            else if (model.model.includes('Microsoft')) familyData.Microsoft.models.push(dataPoint);
            else if (model.model.includes('Mistral')) familyData.Mistral.models.push(dataPoint);
            else familyData.Others.models.push(dataPoint);
        });

        const datasets = Object.keys(familyData)
            .filter(family => familyData[family].models.length > 0)
            .map(family => ({
                label: family,
                data: familyData[family].models,
                backgroundColor: familyData[family].color,
                borderColor: familyData[family].color.replace('0.8', '1'),
                pointRadius: 6,
                pointHoverRadius: 8
            }));

        new Chart(parameterPerformanceCtx, {
            type: 'scatter',
            data: { datasets },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        display: true,
                        position: 'top'
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const point = context.raw;
                                return `${point.model.split('/').pop()}: ${point.x}B params, ${point.y.toFixed(1)}% accuracy`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        type: 'logarithmic',
                        title: { display: true, text: 'Parameter Count (Billions)' },
                        grid: { color: 'rgba(255, 255, 255, 0.1)' }
                    },
                    y: {
                        title: { display: true, text: 'Overall Accuracy (%)' },
                        beginAtZero: true,
                        grid: { color: 'rgba(255, 255, 255, 0.1)' }
                    }
                }
            }
        });
    }


    // Token Efficiency by Family Chart
    const tokenCtx = document.getElementById('tokenEfficiencyChart');
    if (tokenCtx) {
        const families = {
            'OpenAI': rankedModelData.filter(m => m.model.includes('OpenAI')),
            'Google': rankedModelData.filter(m => m.model.includes('Google')),
            'Qwen': rankedModelData.filter(m => m.model.includes('Qwen')),
            'Meta': rankedModelData.filter(m => m.model.includes('Meta')),
            'Microsoft': rankedModelData.filter(m => m.model.includes('Microsoft')),
            'Mistral': rankedModelData.filter(m => m.model.includes('Mistral'))
        };

        const colors = [
            'rgba(102, 126, 234, 0.8)',
            'rgba(240, 147, 251, 0.8)',
            'rgba(16, 185, 129, 0.8)',
            'rgba(245, 158, 11, 0.8)',
            'rgba(239, 68, 68, 0.8)',
            'rgba(139, 92, 246, 0.8)'
        ];

        new Chart(tokenCtx, {
            type: 'bar',
            data: {
                labels: Object.keys(families),
                datasets: [{
                    label: 'Avg Token Efficiency',
                    data: Object.values(families).map(family =>
                        family.length > 0 ? family.reduce((acc, m) => acc + m.efficiency, 0) / family.length : 0
                    ),
                    backgroundColor: colors,
                    borderColor: colors.map(c => c.replace('0.8', '1')),
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    y: {
                        title: { display: true, text: 'Efficiency Score' }
                    }
                }
            }
        });
    }

    // Parameter Trends Chart
    const parameterTrendsCtx = document.getElementById('parameterTrendsChart');
    if (parameterTrendsCtx) {
        // Group models by parameter size categories
        const sizeCategories = {
            'Small (<3B)': rankedModelData.filter(m => parseFloat(m.params) < 3),
            'Medium (3-20B)': rankedModelData.filter(m => parseFloat(m.params) >= 3 && parseFloat(m.params) < 20),
            'Large (20-100B)': rankedModelData.filter(m => parseFloat(m.params) >= 20 && parseFloat(m.params) < 100),
            'XLarge (100B+)': rankedModelData.filter(m => parseFloat(m.params) >= 100)
        };

        const categories = Object.keys(sizeCategories);
        const avgAccuracy = categories.map(cat => {
            const models = sizeCategories[cat];
            return models.length > 0 ? models.reduce((sum, m) => sum + m.accuracy, 0) / models.length : 0;
        });
        const avgInstruction = categories.map(cat => {
            const models = sizeCategories[cat];
            return models.length > 0 ? models.reduce((sum, m) => sum + m.instruction, 0) / models.length : 0;
        });

        new Chart(parameterTrendsCtx, {
            type: 'line',
            data: {
                labels: categories,
                datasets: [
                    {
                        label: 'Average Accuracy',
                        data: avgAccuracy,
                        borderColor: 'rgba(102, 126, 234, 1)',
                        backgroundColor: 'rgba(102, 126, 234, 0.1)',
                        borderWidth: 3,
                        pointRadius: 6,
                        pointHoverRadius: 8,
                        tension: 0.4,
                        fill: true
                    },
                    {
                        label: 'Average Instruction Following',
                        data: avgInstruction,
                        borderColor: 'rgba(240, 147, 251, 1)',
                        backgroundColor: 'rgba(240, 147, 251, 0.1)',
                        borderWidth: 3,
                        pointRadius: 6,
                        pointHoverRadius: 8,
                        tension: 0.4,
                        fill: true
                    }
                ]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        display: true,
                        position: 'top'
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false,
                    }
                },
                interaction: {
                    mode: 'nearest',
                    axis: 'x',
                    intersect: false
                },
                scales: {
                    x: {
                        title: { display: true, text: 'Parameter Size Category' },
                        grid: { color: 'rgba(255, 255, 255, 0.1)' }
                    },
                    y: {
                        title: { display: true, text: 'Performance Score (%)' },
                        beginAtZero: true,
                        max: 100,
                        grid: { color: 'rgba(255, 255, 255, 0.1)' }
                    }
                }
            }
        });
    }

    // Additional charts...
    initInstructionChart();
}

// Initialize instruction following chart
function initInstructionChart() {
    const instructionCtx = document.getElementById('instructionChart');
    if (!instructionCtx) return;

    const buckets = [
        { label: '90-100%', min: 90, max: 100 },
        { label: '80-90%', min: 80, max: 90 },
        { label: '70-80%', min: 70, max: 80 },
        { label: '60-70%', min: 60, max: 70 },
        { label: '<60%', min: 0, max: 60 }
    ];

    const data = buckets.map(bucket => {
        return rankedModelData.filter(m =>
            m.instruction >= bucket.min && m.instruction < bucket.max
        ).length;
    });

    new Chart(instructionCtx, {
        type: 'doughnut',
        data: {
            labels: buckets.map(b => b.label),
            datasets: [{
                data: data,
                backgroundColor: [
                    'rgba(16, 185, 129, 0.8)',
                    'rgba(102, 126, 234, 0.8)',
                    'rgba(245, 158, 11, 0.8)',
                    'rgba(239, 68, 68, 0.8)',
                    'rgba(107, 114, 128, 0.8)'
                ]
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: 'bottom'
                }
            }
        }
    });
}


// Initialize scroll animations
function initScrollAnimations() {
    const observer = new IntersectionObserver((entries) => {
        entries.forEach((entry) => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
            }
        });
    });

    document.querySelectorAll('.fade-in, .slide-in-left, .slide-in-right').forEach((el) => {
        observer.observe(el);
    });
}

// FAB functionality
function initFAB() {
    const fabBtn = document.getElementById('fabBtn');
    const fabMenu = document.getElementById('fabMenu');

    if (fabBtn && fabMenu) {
        fabBtn.addEventListener('click', () => {
            fabMenuOpen = !fabMenuOpen;
            fabMenu.classList.toggle('active', fabMenuOpen);
            fabBtn.querySelector('i').className = fabMenuOpen ? 'fas fa-times' : 'fas fa-plus';
        });
    }
}

// Export functionality
function exportData() {
    const csvContent = generateCSV(filteredData);
    downloadCSV(csvContent, 'beyondbench-leaderboard.csv');
}

function generateCSV(data) {
    const headers = ['Rank', 'Model', 'Parameters', 'Accuracy', 'Efficiency', 'Instruction', 'Tokens', 'Words', 'Characters'];
    const csvRows = [headers.join(',')];

    data.forEach(model => {
        const row = [
            model.rank,
            `"${model.model}"`,
            model.params,
            model.accuracy.toFixed(2),
            model.efficiency.toFixed(3),
            model.instruction.toFixed(1),
            model.tokens.toFixed(0),
            model.words.toFixed(0),
            model.chars.toFixed(0)
        ];
        csvRows.push(row.join(','));
    });

    return csvRows.join('\n');
}

function downloadCSV(content, fileName) {
    const blob = new Blob([content], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = fileName;
    a.click();
    window.URL.revokeObjectURL(url);
}

// Utility functions
function scrollToTop() {
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

function shareResults() {
    if (navigator.share) {
        navigator.share({
            title: 'BeyondBench Leaderboard',
            text: 'Check out the latest language model reasoning benchmarks!',
            url: window.location.href
        });
    } else {
        // Fallback
        navigator.clipboard.writeText(window.location.href).then(() => {
            alert('URL copied to clipboard!');
        });
    }
}

// Modal functionality
function initModals() {
    const modals = document.querySelectorAll('.modal');
    const closeButtons = document.querySelectorAll('.close');

    closeButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            btn.closest('.modal').classList.remove('active');
        });
    });

    modals.forEach(modal => {
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                modal.classList.remove('active');
            }
        });
    });
}

// Initialize everything
document.addEventListener('DOMContentLoaded', function () {
    console.log('BeyondBench: JavaScript is loading...', rankedModelData.length, 'models found');
    // Set initial filtered data
    filteredData = [...rankedModelData];

    // Initialize theme first
    initTheme();

    // Initialize all components
    createParticles();
    generateAuthorInfo();
    updateTopModels();
    initFilters();
    initScrollAnimations();
    initFAB();
    initModals();

    // Render initial table
    renderTable(filteredData);
    updateStats();

    // Initialize charts
    setTimeout(initCharts, 100); // Delay to ensure canvas elements are ready

    // Add smooth scrolling for navigation links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const targetId = this.getAttribute('href');
            const targetElement = document.querySelector(targetId);
            if (targetElement) {
                targetElement.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });

    // Add export button functionality
    const exportBtn = document.getElementById('exportBtn');
    if (exportBtn) {
        exportBtn.addEventListener('click', exportData);
    }

    // Add compare button functionality
    const compareBtn = document.getElementById('compareBtn');
    if (compareBtn) {
        compareBtn.addEventListener('click', () => {
            if (selectedModels.size > 0) {
                showComparison();
            }
        });
    }

    console.log('BeyondBench Leaderboard initialized successfully!');
});

// Show model comparison
function showComparison() {
    const modal = document.getElementById('comparisonModal');
    const content = document.getElementById('comparisonContent');

    if (!modal || !content) return;

    const models = Array.from(selectedModels).map(id => rankedModelData.find(m => m.model === id)).filter(Boolean);

    content.innerHTML = `
        <div class="comparison-grid">
            ${models.map(model => `
                <div class="comparison-card">
                    <h3>${model.model.split('/').pop()}</h3>
                    <div class="comparison-metrics">
                        <div class="metric">
                            <span class="label">Rank</span>
                            <span class="value rank-${model.rank <= 3 ? model.rank : ''}">#${model.rank}</span>
                        </div>
                        <div class="metric">
                            <span class="label">Accuracy</span>
                            <span class="value">${model.accuracy.toFixed(1)}%</span>
                        </div>
                        <div class="metric">
                            <span class="label">Efficiency</span>
                            <span class="value">${model.efficiency.toFixed(3)}</span>
                        </div>
                        <div class="metric">
                            <span class="label">Instruction</span>
                            <span class="value">${model.instruction.toFixed(1)}%</span>
                        </div>
                        <div class="metric">
                            <span class="label">Tokens</span>
                            <span class="value">${model.tokens.toFixed(0)}</span>
                        </div>
                    </div>
                </div>
            `).join('')}
        </div>
        <div class="comparison-actions">
            <button class="btn btn-secondary" onclick="clearSelection()">Clear Selection</button>
            <button class="btn btn-primary" onclick="exportComparison()">Export Comparison</button>
        </div>
    `;

    modal.classList.add('active');
}

function clearSelection() {
    selectedModels.clear();
    updateCompareButton();
    document.getElementById('comparisonModal').classList.remove('active');
}

function exportComparison() {
    const models = Array.from(selectedModels).map(id => rankedModelData.find(m => m.model === id)).filter(Boolean);
    const csvContent = generateCSV(models);
    downloadCSV(csvContent, 'model-comparison.csv');
}

// Theme Toggle Functionality
function initTheme() {
    const themeToggle = document.getElementById('themeToggle');
    const themeIcon = document.getElementById('themeIcon');
    const html = document.documentElement;

    // Check for saved theme preference or default to dark
    const currentTheme = localStorage.getItem('theme') || 'dark';

    // Set initial theme
    if (currentTheme === 'light') {
        html.setAttribute('data-theme', 'light');
        themeIcon.className = 'fas fa-sun';
    } else {
        html.setAttribute('data-theme', 'dark');
        themeIcon.className = 'fas fa-moon';
    }

    // Theme toggle event listener
    themeToggle.addEventListener('click', () => {
        const currentTheme = html.getAttribute('data-theme');
        const newTheme = currentTheme === 'dark' ? 'light' : 'dark';

        html.setAttribute('data-theme', newTheme);
        localStorage.setItem('theme', newTheme);

        // Update icon with animation
        themeIcon.style.transform = 'rotate(360deg)';
        setTimeout(() => {
            themeIcon.className = newTheme === 'dark' ? 'fas fa-moon' : 'fas fa-sun';
            themeIcon.style.transform = 'rotate(0deg)';
        }, 150);

        // Refresh charts with new theme
        setTimeout(() => {
            // Clear existing charts and reinitialize
            Chart.helpers.each(Chart.instances, function(instance){
                instance.destroy();
            });
            initCharts();
        }, 300);
    });
}