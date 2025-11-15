// Main JavaScript functionality for Power Grid Analysis System

// Global notification system
function showNotification(message, type = 'info') {
    const toast = document.getElementById('notificationToast');
    const toastMessage = document.getElementById('toastMessage');
    
    if (!toast || !toastMessage) return;
    
    // Set message and style based on type
    toastMessage.textContent = message;
    
    // Remove existing classes
    toast.classList.remove('bg-success', 'bg-danger', 'bg-warning', 'bg-info');
    
    // Add appropriate class based on type
    switch (type) {
        case 'success':
            toast.classList.add('bg-success');
            break;
        case 'danger':
        case 'error':
            toast.classList.add('bg-danger');
            break;
        case 'warning':
            toast.classList.add('bg-warning');
            break;
        default:
            toast.classList.add('bg-info');
            break;
    }
    
    // Show the toast
    const bsToast = new bootstrap.Toast(toast, {
        autohide: true,
        delay: 5000
    });
    bsToast.show();
}

// Global loading state management
function setLoadingState(element, isLoading, originalText = null) {
    if (!element) return;
    
    if (isLoading) {
        element.dataset.originalText = originalText || element.innerHTML;
        element.innerHTML = '<i class="fas fa-spinner fa-spin me-1"></i>Loading...';
        element.disabled = true;
    } else {
        element.innerHTML = element.dataset.originalText || originalText || element.innerHTML;
        element.disabled = false;
        delete element.dataset.originalText;
    }
}

// API error handling
function handleApiError(error, context = 'operation') {
    console.error(`Error in ${context}:`, error);
    
    if (error.message) {
        showNotification(`Error: ${error.message}`, 'danger');
    } else {
        showNotification(`An error occurred during ${context}`, 'danger');
    }
}

// Format numbers for display
function formatNumber(num, decimals = 0) {
    if (num === null || num === undefined) return 'N/A';
    return Number(num).toLocaleString(undefined, {
        minimumFractionDigits: decimals,
        maximumFractionDigits: decimals
    });
}

// Format percentage
function formatPercentage(value, decimals = 1) {
    if (value === null || value === undefined) return 'N/A';
    return `${(value * 100).toFixed(decimals)}%`;
}

// Format date/time
function formatDateTime(dateString) {
    if (!dateString) return 'N/A';
    try {
        return new Date(dateString).toLocaleString('en-IN', {
            timeZone: 'Asia/Kolkata',
            year: 'numeric',
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit'
        });
    } catch (error) {
        return 'Invalid Date';
    }
}

// Debounce function for search/filter inputs
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Copy text to clipboard
async function copyToClipboard(text) {
    try {
        await navigator.clipboard.writeText(text);
        showNotification('Copied to clipboard', 'success');
    } catch (error) {
        console.error('Failed to copy to clipboard:', error);
        showNotification('Failed to copy to clipboard', 'danger');
    }
}

// Download data as JSON
function downloadJSON(data, filename = 'data.json') {
    try {
        const blob = new Blob([JSON.stringify(data, null, 2)], {
            type: 'application/json'
        });
        const url = URL.createObjectURL(blob);
        
        const link = document.createElement('a');
        link.href = url;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        
        URL.revokeObjectURL(url);
        showNotification('Data downloaded successfully', 'success');
    } catch (error) {
        console.error('Download failed:', error);
        showNotification('Download failed', 'danger');
    }
}

// Download data as CSV
function downloadCSV(data, filename = 'data.csv') {
    try {
        if (!Array.isArray(data) || data.length === 0) {
            showNotification('No data to download', 'warning');
            return;
        }
        
        // Get headers from first object
        const headers = Object.keys(data[0]);
        
        // Create CSV content
        let csv = headers.join(',') + '\n';
        
        data.forEach(row => {
            const values = headers.map(header => {
                const value = row[header];
                // Escape commas and quotes
                if (typeof value === 'string' && (value.includes(',') || value.includes('"'))) {
                    return `"${value.replace(/"/g, '""')}"`;
                }
                return value;
            });
            csv += values.join(',') + '\n';
        });
        
        const blob = new Blob([csv], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        
        const link = document.createElement('a');
        link.href = url;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        
        URL.revokeObjectURL(url);
        showNotification('CSV downloaded successfully', 'success');
    } catch (error) {
        console.error('CSV download failed:', error);
        showNotification('CSV download failed', 'danger');
    }
}

// Initialize tooltips
function initializeTooltips() {
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

// Initialize popovers
function initializePopovers() {
    const popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    popoverTriggerList.map(function (popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });
}

// Animate numbers (counting effect)
function animateNumber(element, start, end, duration = 1000) {
    if (!element) return;
    
    const range = end - start;
    const increment = range / (duration / 16); // 60 FPS
    let current = start;
    
    const timer = setInterval(() => {
        current += increment;
        if ((increment > 0 && current >= end) || (increment < 0 && current <= end)) {
            current = end;
            clearInterval(timer);
        }
        element.textContent = Math.round(current);
    }, 16);
}

// Validate form data
function validateForm(formData, rules) {
    const errors = [];
    
    for (const [field, rule] of Object.entries(rules)) {
        const value = formData.get(field);
        
        if (rule.required && (!value || value.trim() === '')) {
            errors.push(`${rule.label || field} is required`);
            continue;
        }
        
        if (value && rule.type === 'number') {
            const num = parseFloat(value);
            if (isNaN(num)) {
                errors.push(`${rule.label || field} must be a valid number`);
                continue;
            }
            
            if (rule.min !== undefined && num < rule.min) {
                errors.push(`${rule.label || field} must be at least ${rule.min}`);
            }
            
            if (rule.max !== undefined && num > rule.max) {
                errors.push(`${rule.label || field} must be at most ${rule.max}`);
            }
        }
        
        if (value && rule.pattern && !rule.pattern.test(value)) {
            errors.push(`${rule.label || field} has invalid format`);
        }
    }
    
    return errors;
}

// Initialize theme handling
function initializeTheme() {
    // The app uses dark theme by default
    document.documentElement.setAttribute('data-bs-theme', 'dark');
}

// Handle window resize events
function handleResize() {
    // Trigger resize event for charts and visualizations
    window.dispatchEvent(new Event('resize'));
}

// Keyboard shortcuts
function initializeKeyboardShortcuts() {
    document.addEventListener('keydown', (event) => {
        // Ctrl/Cmd + K for search (if search functionality exists)
        if ((event.ctrlKey || event.metaKey) && event.key === 'k') {
            event.preventDefault();
            const searchInput = document.querySelector('input[type="search"]');
            if (searchInput) {
                searchInput.focus();
            }
        }
        
        // Escape to close modals/dropdowns
        if (event.key === 'Escape') {
            const modals = document.querySelectorAll('.modal.show');
            modals.forEach(modal => {
                const bsModal = bootstrap.Modal.getInstance(modal);
                if (bsModal) {
                    bsModal.hide();
                }
            });
        }
    });
}

// Initialize on DOM content loaded
document.addEventListener('DOMContentLoaded', function() {
    initializeTooltips();
    initializePopovers();
    initializeTheme();
    initializeKeyboardShortcuts();
    window.addEventListener('resize', debounce(handleResize, 250));
    
    const forms = document.querySelectorAll('form');
    forms.forEach(form => {
        form.addEventListener('submit', function(event) {
            const submitBtn = form.querySelector('button[type="submit"]');
            if (submitBtn) {
                setLoadingState(submitBtn, true);
                setTimeout(() => {
                    setLoadingState(submitBtn, false);
                }, 10000);
            }
        });
    });
    
    document.addEventListener('click', function(event) {
        if (event.target.matches('[data-copy]')) {
            const textToCopy = event.target.getAttribute('data-copy');
            copyToClipboard(textToCopy);
        }
    });
    
    console.log('Power Grid Analysis System initialized');
});

// Global error handler
window.addEventListener('error', function(event) {
    console.error('Global error:', event.error);
    showNotification('An unexpected error occurred', 'danger');
});

// Handle unhandled promise rejections
window.addEventListener('unhandledrejection', function(event) {
    console.error('Unhandled promise rejection:', event.reason);
    showNotification('An unexpected error occurred', 'danger');
});
