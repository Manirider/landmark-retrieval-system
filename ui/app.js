/**
 * Landmark Retrieval System Frontend Controller
 */
document.addEventListener('DOMContentLoaded', () => {
    // State management
    const state = {
        selectedFile: null,
        apiUrl: localStorage.getItem('landmark_api_url') || 'http://localhost:8000',
        demoMode: true, // Default to true so it works out of the box on Vercel
        healthPollInterval: null,
        isAnalyzing: false
    };

    // DOM Elements
    const demoToggle = document.getElementById('demo-mode-toggle');
    const apiStatusBadge = document.getElementById('api-status-badge');
    const apiStatusText = document.getElementById('api-status-text');
    const settingsToggle = document.getElementById('settings-toggle');
    const settingsContent = document.getElementById('settings-content');
    const apiUrlInput = document.getElementById('api-url-input');
    const apiTestBtn = document.getElementById('api-test-btn');
    const dropZone = document.getElementById('drop-zone');
    const imageFileInput = document.getElementById('image-file-input');
    const dropZonePrompt = document.getElementById('drop-zone-prompt');
    const dropZonePreview = document.getElementById('drop-zone-preview');
    const previewImage = document.getElementById('preview-image');
    const removePreviewBtn = document.getElementById('remove-preview-btn');
    const uploadForm = document.getElementById('upload-form');
    const analyzeBtn = document.getElementById('analyze-btn');
    const analyzeBtnSpinner = analyzeBtn.querySelector('.btn-spinner');
    const analyzeBtnText = analyzeBtn.querySelector('span');
    
    // Results panels
    const resultsCount = document.getElementById('results-count');
    const resultsPlaceholder = document.getElementById('results-placeholder');
    const resultsLoading = document.getElementById('results-loading');
    const resultsError = document.getElementById('results-error');
    const resultsList = document.getElementById('results-list');
    const errorTitle = document.getElementById('error-title');
    const errorMsg = document.getElementById('error-msg');
    const retryBtn = document.getElementById('retry-btn');

    // Preset mock landmarks list for Demo Mode simulation
    const mockLandmarks = [
        [
            { landmark_name: "Taj Mahal", score: 0.954 },
            { landmark_name: "Humayun's Tomb", score: 0.612 },
            { landmark_name: "Qutub Minar", score: 0.384 },
            { landmark_name: "Red Fort", score: 0.221 },
            { landmark_name: "India Gate", score: 0.108 }
        ],
        [
            { landmark_name: "Eiffel Tower", score: 0.967 },
            { landmark_name: "Arc de Triomphe", score: 0.723 },
            { landmark_name: "Louvre Museum", score: 0.449 },
            { landmark_name: "Sacré-Cœur", score: 0.281 },
            { landmark_name: "Notre-Dame Cathedral", score: 0.154 }
        ],
        [
            { landmark_name: "Colosseum", score: 0.941 },
            { landmark_name: "Roman Forum", score: 0.678 },
            { landmark_name: "Pantheon", score: 0.492 },
            { landmark_name: "Trevi Fountain", score: 0.315 },
            { landmark_name: "St. Peter's Basilica", score: 0.198 }
        ],
        [
            { landmark_name: "Machu Picchu", score: 0.982 },
            { landmark_name: "Sacsayhuamán", score: 0.548 },
            { landmark_name: "Ollantaytambo", score: 0.324 },
            { landmark_name: "Coricancha", score: 0.187 },
            { landmark_name: "Pisac Ruins", score: 0.092 }
        ],
        [
            { landmark_name: "Statue of Liberty", score: 0.958 },
            { landmark_name: "Ellis Island", score: 0.691 },
            { landmark_name: "Empire State Building", score: 0.412 },
            { landmark_name: "Brooklyn Bridge", score: 0.276 },
            { landmark_name: "Chrysler Building", score: 0.143 }
        ]
    ];

    // Initialize UI settings
    apiUrlInput.value = state.apiUrl;
    demoToggle.checked = state.demoMode;

    // Toggle Settings Panel
    settingsToggle.addEventListener('click', () => {
        settingsContent.classList.toggle('active');
        settingsToggle.classList.toggle('collapsed');
    });

    // Handle API URL Change
    apiUrlInput.addEventListener('input', (e) => {
        state.apiUrl = e.target.value.trim().replace(/\/$/, ""); // Strip trailing slash
        localStorage.setItem('landmark_api_url', state.apiUrl);
        if (!state.demoMode) {
            checkBackendHealth();
        }
    });

    // Test API Button
    apiTestBtn.addEventListener('click', () => {
        const originalText = apiTestBtn.textContent;
        apiTestBtn.disabled = true;
        apiTestBtn.textContent = 'Testing...';
        
        checkBackendHealth().finally(() => {
            apiTestBtn.disabled = false;
            apiTestBtn.textContent = originalText;
        });
    });

    // Toggle Demo Mode
    demoToggle.addEventListener('change', (e) => {
        state.demoMode = e.target.checked;
        updateBadgeStatus();
    });

    // Update Status Badge visually
    function updateBadgeStatus(status = 'checking') {
        apiStatusBadge.className = 'status-badge';
        
        if (state.demoMode) {
            apiStatusBadge.classList.add('online');
            apiStatusBadge.style.borderColor = 'rgba(168, 85, 247, 0.3)'; // Violet border
            apiStatusBadge.style.background = 'rgba(168, 85, 247, 0.1)';
            apiStatusText.textContent = 'Demo Mode Active';
            apiStatusText.style.color = '#a855f7';
            const dot = apiStatusBadge.querySelector('.status-indicator');
            if (dot) {
                dot.style.backgroundColor = '#a855f7';
                dot.style.boxShadow = '0 0 8px #a855f7';
            }
        } else {
            // Restore default styles (which are in CSS)
            apiStatusBadge.removeAttribute('style');
            apiStatusText.removeAttribute('style');
            const dot = apiStatusBadge.querySelector('.status-indicator');
            if (dot) dot.removeAttribute('style');

            if (status === 'online') {
                apiStatusBadge.classList.add('online');
                apiStatusText.textContent = 'Backend Online';
            } else if (status === 'offline') {
                apiStatusBadge.classList.add('offline');
                apiStatusText.textContent = 'Backend Offline';
            } else {
                apiStatusBadge.classList.add('checking');
                apiStatusText.textContent = 'Checking Status';
            }
        }
    }

    // Health check API fetch
    function checkBackendHealth() {
        if (state.demoMode) {
            updateBadgeStatus();
            return Promise.resolve(true);
        }
        
        updateBadgeStatus('checking');
        
        const healthUrl = `${state.apiUrl}/health`;
        return fetch(healthUrl)
            .then(response => {
                if (response.ok) {
                    updateBadgeStatus('online');
                    return true;
                } else {
                    updateBadgeStatus('offline');
                    return false;
                }
            })
            .catch(() => {
                updateBadgeStatus('offline');
                return false;
            });
    }

    // Start polling health
    function startHealthPolling() {
        checkBackendHealth();
        state.healthPollInterval = setInterval(checkBackendHealth, 15000);
    }

    // Stop polling health
    function stopHealthPolling() {
        if (state.healthPollInterval) {
            clearInterval(state.healthPollInterval);
        }
    }

    // Start background check immediately
    startHealthPolling();

    // Trigger File Input Click on Drop Zone Click
    dropZone.addEventListener('click', (e) => {
        if (e.target !== removePreviewBtn && !removePreviewBtn.contains(e.target)) {
            imageFileInput.click();
        }
    });

    // File Input change
    imageFileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });

    // Drag and drop event listeners
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.remove('dragover');
        }, false);
    });

    dropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files && files.length > 0) {
            handleFileSelect(files[0]);
        }
    });

    // Validate and load image
    function handleFileSelect(file) {
        // Validation check
        if (!file.type.match('image/jpeg') && !file.type.match('image/png')) {
            alert('Unsupported format. Please select a JPEG or PNG image.');
            return;
        }

        // 10MB Limit Check
        const maxSize = 10 * 1024 * 1024;
        if (file.size > maxSize) {
            alert('File exceeds the 10MB size limit. Please select a smaller photo.');
            return;
        }

        state.selectedFile = file;

        // Display Image Preview
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
            dropZonePrompt.style.display = 'none';
            dropZonePreview.style.display = 'flex';
            analyzeBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    // Remove Preview Click handler
    removePreviewBtn.addEventListener('click', (e) => {
        e.preventDefault();
        e.stopPropagation();
        resetUploadState();
    });

    function resetUploadState() {
        state.selectedFile = null;
        imageFileInput.value = '';
        previewImage.src = '';
        dropZonePreview.style.display = 'none';
        dropZonePrompt.style.display = 'flex';
        analyzeBtn.disabled = true;
    }

    // Submit Form
    uploadForm.addEventListener('submit', (e) => {
        e.preventDefault();
        if (!state.selectedFile || state.isAnalyzing) return;

        performLandmarkAnalysis();
    });

    // Retry Button Click
    retryBtn.addEventListener('click', () => {
        if (!state.selectedFile) return;
        performLandmarkAnalysis();
    });

    // Main Analysis Trigger
    function performLandmarkAnalysis() {
        state.isAnalyzing = true;
        toggleButtonLoading(true);
        showResultState('loading');

        if (state.demoMode) {
            // Simulate neural network latency (1200ms)
            setTimeout(() => {
                const landmarkNames = {
                    "10000": "Eiffel Tower",
                    "10001": "Statue of Liberty",
                    "10002": "Taj Mahal",
                    "10003": "Great Wall of China",
                    "10004": "Colosseum",
                    "10005": "Machu Picchu",
                    "10006": "Christ the Redeemer",
                    "10007": "Petra",
                    "10008": "Great Pyramid of Giza",
                    "10009": "Angkor Wat",
                    "10010": "Sagrada Familia",
                    "10011": "Big Ben",
                    "10012": "Sydney Opera House",
                    "10013": "Golden Gate Bridge",
                    "10014": "Burj Khalifa",
                    "10015": "Tower of Pisa",
                    "10016": "Hagia Sophia",
                    "10017": "Stonehenge",
                    "10018": "Mount Rushmore",
                    "10019": "Neuschwanstein Castle"
                };

                const fileName = state.selectedFile ? state.selectedFile.name.toLowerCase() : "";
                let matchedId = null;
                
                // 1. Try to find ID match in filename (e.g. "10004_0012.jpg")
                const idMatch = fileName.match(/100\d{2}/);
                if (idMatch && landmarkNames[idMatch[0]]) {
                    matchedId = idMatch[0];
                }
                
                // 2. Try to find keyword match
                if (!matchedId) {
                    for (const [id, name] of Object.entries(landmarkNames)) {
                        const words = name.toLowerCase().split(/\s+/);
                        const match = words.some(word => word.length > 3 && fileName.includes(word));
                        if (match) {
                            matchedId = id;
                            break;
                        }
                    }
                }
                
                // 3. Fallback to deterministic hash index
                let primaryName = "";
                if (matchedId) {
                    primaryName = landmarkNames[matchedId];
                } else {
                    let hash = 0;
                    for (let i = 0; i < fileName.length; i++) {
                        hash = fileName.charCodeAt(i) + ((hash << 5) - hash);
                    }
                    const keys = Object.keys(landmarkNames);
                    const fallbackId = keys[Math.abs(hash) % keys.length];
                    primaryName = landmarkNames[fallbackId];
                }
                
                // Build dynamic results
                const otherNames = Object.values(landmarkNames).filter(n => n !== primaryName);
                // Shuffle other names deterministically based on primary name
                let nameHash = 0;
                for (let i = 0; i < primaryName.length; i++) {
                    nameHash += primaryName.charCodeAt(i);
                }
                
                const seedRandom = (seed) => {
                    const x = Math.sin(seed++) * 10000;
                    return x - Math.floor(x);
                };
                
                for (let i = otherNames.length - 1; i > 0; i--) {
                    const j = Math.floor(seedRandom(nameHash + i) * (i + 1));
                    [otherNames[i], otherNames[j]] = [otherNames[j], otherNames[i]];
                }
                
                const mockResults = [
                    { landmark_name: primaryName, score: 0.92 + seedRandom(nameHash) * 0.07 },
                    { landmark_name: otherNames[0], score: 0.55 + seedRandom(nameHash + 1) * 0.15 },
                    { landmark_name: otherNames[1], score: 0.30 + seedRandom(nameHash + 2) * 0.15 },
                    { landmark_name: otherNames[2], score: 0.15 + seedRandom(nameHash + 3) * 0.12 },
                    { landmark_name: otherNames[3], score: 0.05 + seedRandom(nameHash + 4) * 0.08 }
                ];
                
                renderResults(mockResults);
                state.isAnalyzing = false;
                toggleButtonLoading(false);
            }, 1200);
        } else {
            // Actual API Call
            const formData = new FormData();
            formData.append('image', state.selectedFile);

            fetch(`${state.apiUrl}/retrieve`, {
                method: 'POST',
                body: formData
            })
            .then(async (response) => {
                if (!response.ok) {
                    const errData = await response.json().catch(() => ({ detail: 'Analysis failed' }));
                    throw new Error(errData.detail || 'Internal server retrieval error');
                }
                return response.json();
            })
            .then((data) => {
                if (data && data.length > 0) {
                    renderResults(data);
                } else {
                    showResultError('No Landmarks Found', 'The system completed search successfully but could not match the features in the index.');
                }
            })
            .catch((err) => {
                showResultError('API Error', err.message || 'Could not establish connection to the FastAPI server. Please verify your API URL configuration or toggle Demo Mode.');
            })
            .finally(() => {
                state.isAnalyzing = false;
                toggleButtonLoading(false);
            });
        }
    }

    // Enable/disable Analyze Button
    function toggleButtonLoading(isLoading) {
        if (isLoading) {
            analyzeBtn.disabled = true;
            analyzeBtnSpinner.style.display = 'inline-block';
            analyzeBtnText.textContent = 'Processing Vector...';
        } else {
            analyzeBtn.disabled = false;
            analyzeBtnSpinner.style.display = 'none';
            analyzeBtnText.textContent = 'Analyze Image';
        }
    }

    // Results states helper
    function showResultState(stateName) {
        resultsPlaceholder.style.display = 'none';
        resultsLoading.style.display = 'none';
        resultsError.style.display = 'none';
        resultsList.style.display = 'none';
        resultsCount.style.display = 'none';

        if (stateName === 'placeholder') {
            resultsPlaceholder.style.display = 'flex';
        } else if (stateName === 'loading') {
            resultsLoading.style.display = 'flex';
        } else if (stateName === 'error') {
            resultsError.style.display = 'flex';
        } else if (stateName === 'results') {
            resultsList.style.display = 'flex';
            resultsCount.style.display = 'block';
        }
    }

    // Error states injector
    function showResultError(title, message) {
        errorTitle.textContent = title;
        errorMsg.textContent = message;
        showResultState('error');
    }

    // Results Render Loop
    function renderResults(results) {
        resultsList.innerHTML = '';
        resultsCount.textContent = `${results.length} matches found`;
        
        results.forEach((item, index) => {
            const percentage = (item.score * 100).toFixed(1);
            const rank = index + 1;
            
            const resultElement = document.createElement('div');
            resultElement.className = 'result-item';
            
            resultElement.innerHTML = `
                <div class="result-meta">
                    <div class="result-rank-name">
                        <span class="result-rank">${rank}</span>
                        <span class="result-name">${item.landmark_name}</span>
                    </div>
                    <span class="result-score">${percentage}%</span>
                </div>
                <div class="result-progress-track">
                    <div class="result-progress-bar" id="bar-${rank}"></div>
                </div>
            `;
            
            resultsList.appendChild(resultElement);
            
            // Short delay to trigger progress bar loading width transition
            setTimeout(() => {
                const bar = document.getElementById(`bar-${rank}`);
                if (bar) {
                    bar.style.width = `${percentage}%`;
                }
            }, 100 + (index * 50));
        });

        showResultState('results');
    }

    // Handle clean termination
    window.addEventListener('beforeunload', stopHealthPolling);
});
