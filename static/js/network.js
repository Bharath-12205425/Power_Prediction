// Network visualization functionality
class NetworkVisualization {
    constructor() {
        this.svg = null;
        this.simulation = null;
        this.nodes = [];
        this.links = [];
        this.selectedNode = null;
        this.width = 800;
        this.height = 600;
        this.colorScale = d3.scaleOrdinal(d3.schemeCategory10);
    }

    init() {
        this.initializeSVG();
        this.loadNetworkData();
    }

    initializeSVG() {
        const container = document.getElementById('networkContainer');
        this.width = container.clientWidth;
        this.height = container.clientHeight;

        this.svg = d3.select('#networkSvg')
            .attr('width', this.width)
            .attr('height', this.height);

        // Add zoom behavior
        const zoom = d3.zoom()
            .scaleExtent([0.1, 4])
            .on('zoom', (event) => {
                this.svg.select('.network-group')
                    .attr('transform', event.transform);
            });

        this.svg.call(zoom);

        // Create main group for network elements
        this.svg.append('g').attr('class', 'network-group');
    }

    async loadNetworkData() {
        document.getElementById('networkLoading').style.display = 'block';
        
        try {
            const response = await fetch('/api/network-data');
            const data = await response.json();
            
            if (data.success) {
                this.nodes = data.data.nodes || [];
                this.links = data.data.edges || [];
                // Load GNN predictions (if available) and attach to nodes
                await this.loadGnnPredictions();

                this.renderNetwork();
                this.updateLegend();
                this.updateNetworkStats(data.data.stats);
            } else {
                console.error('Failed to load network data:', data.error);
                this.showError('Failed to load network data');
            }
        } catch (error) {
            console.error('Error loading network data:', error);
            this.showError('Error loading network data');
        } finally {
            document.getElementById('networkLoading').style.display = 'none';
        }
    }

    async loadGnnPredictions() {
        try {
            const res = await fetch('/api/gnn/predictions');
            const data = await res.json();

            if (data && data.success) {
                // Attach GNN probabilities to each node by id
                this.nodes.forEach(node => {
                    node.gnn_probability = (data.data && data.data[node.id]) ? data.data[node.id] : (node.gnn_probability || 0);
                });
            }
        } catch (error) {
            console.error("Failed to load GNN predictions", error);
        }
    }

    renderNetwork() {
        if (this.nodes.length === 0) {
            this.showError('No network data available');
            return;
        }

        // Clear existing elements
        this.svg.select('.network-group').selectAll('*').remove();

        // Create force simulation
        this.simulation = d3.forceSimulation(this.nodes)
            .force('link', d3.forceLink(this.links).id(d => d.id).distance(100))
            .force('charge', d3.forceManyBody().strength(-300))
            .force('center', d3.forceCenter(this.width / 2, this.height / 2))
            .force('collision', d3.forceCollide().radius(30));

        // Create links
        const linkGroup = this.svg.select('.network-group')
            .append('g')
            .attr('class', 'links');

        const links = linkGroup.selectAll('line')
            .data(this.links)
            .enter().append('line')
            .attr('class', 'network-link')
            .style('stroke', '#666')
            .style('stroke-opacity', 0.6)
            .style('stroke-width', 2);

        // Create nodes
        const nodeGroup = this.svg.select('.network-group')
            .append('g')
            .attr('class', 'nodes');

        const nodes = nodeGroup.selectAll('circle')
            .data(this.nodes)
            .enter().append('circle')
            .attr('class', 'network-node')
            .attr('r', d => this.getNodeRadius(d))
            .style('fill', d => this.getNodeColor(d))
            .style('stroke', '#fff')
            .style('stroke-width', 2)
            .style('cursor', 'pointer')
            .call(this.createDragBehavior())
            .on('click', (event, d) => this.selectNode(d))
            .on('mouseover', (event, d) => this.showNodeTooltip(event, d))
            .on('mouseout', () => this.hideNodeTooltip());

        // Create labels
        const labelGroup = this.svg.select('.network-group')
            .append('g')
            .attr('class', 'labels');

        const labels = labelGroup.selectAll('text')
            .data(this.nodes)
            .enter().append('text')
            .attr('class', 'network-label')
            .text(d => d.id)
            .style('font-size', '12px')
            .style('fill', '#fff')
            .style('text-anchor', 'middle')
            .style('pointer-events', 'none');

        // Update positions on simulation tick
        this.simulation.on('tick', () => {
            links
                .attr('x1', d => d.source.x)
                .attr('y1', d => d.source.y)
                .attr('x2', d => d.target.x)
                .attr('y2', d => d.target.y);

            nodes
                .attr('cx', d => d.x)
                .attr('cy', d => d.y);

            labels
                .attr('x', d => d.x)
                .attr('y', d => d.y + 4);
        });
    }

    getNodeRadius(node) {
        const sizeBy = document.getElementById('nodeSize').value;
        
        switch (sizeBy) {
            case 'degree':
                return Math.max(8, Math.min(25, 8 + (node.degree || 0) * 2));
            case 'capacity':
                const capacity = node.max_capacity_kwh || 2000;
                return Math.max(8, Math.min(25, 8 + (capacity / 500)));
            case 'centrality':
                const centrality = node.degree_centrality || 0;
                return Math.max(8, Math.min(25, 8 + centrality * 20));
            default:
                return 12;
        }
    }

    // Return effective probability to use for coloring: prefer GNN unless it's 0 (treated as missing),
    // then fallback to RF, XGB/GBDT, or the legacy current_risk_score.
    getEffectiveGnnProbability(node) {
        const gnn = (node.gnn_probability === undefined) ? null : node.gnn_probability;

        if (gnn && gnn > 0) {
            return { prob: gnn, source: 'GNN' };
        }

        // Fallback order: rf_probability, xgb_probability, gbdt_probability, current_risk_score
        if (node.rf_probability !== undefined && node.rf_probability !== null) {
            return { prob: node.rf_probability, source: 'RandomForest' };
        }

        if (node.xgb_probability !== undefined && node.xgb_probability !== null) {
            return { prob: node.xgb_probability, source: 'XGBoost' };
        }

        if (node.gbdt_probability !== undefined && node.gbdt_probability !== null) {
            return { prob: node.gbdt_probability, source: 'GBDT' };
        }

        // Last resort: use the legacy current_risk_score
        return { prob: (node.current_risk_score || 0), source: 'Legacy' };
    }

    getNodeColor(node) {
        const colorBy = document.getElementById('colorBy').value;
        
        switch (colorBy) {
            case 'building_type':
                return this.colorScale(node.building_type || 'Unknown');
            case 'risk_level':
                const risk = node.current_risk_score || 0;
                if (risk > 0.7) return '#dc3545';
                if (risk > 0.4) return '#ffc107';
                return '#198754';
            case 'gnn_risk':
                const eff = this.getEffectiveGnnProbability(node);
                const p_use = eff.prob || 0;
                if (p_use > 0.7) return '#dc3545';
                if (p_use > 0.4) return '#ffc107';
                return '#198754';
            case 'centrality':
                const centrality = node.degree_centrality || 0;
                const hue = (1 - centrality) * 240; // Blue to red
                return d3.hsl(hue, 0.8, 0.6);
            case 'capacity':
                const capacity = node.max_capacity_kwh || 2000;
                const scale = d3.scaleLinear()
                    .domain([1000, 6000])
                    .range(['#17a2b8', '#fd7e14']);
                return scale(capacity);
            default:
                return '#007bff';
        }
    }

    createDragBehavior() {
        return d3.drag()
            .on('start', (event, d) => {
                if (!event.active) this.simulation.alphaTarget(0.3).restart();
                d.fx = d.x;
                d.fy = d.y;
            })
            .on('drag', (event, d) => {
                d.fx = event.x;
                d.fy = event.y;
            })
            .on('end', (event, d) => {
                if (!event.active) this.simulation.alphaTarget(0);
                d.fx = null;
                d.fy = null;
            });
    }

    selectNode(node) {
        // Highlight selected node
        this.svg.selectAll('.network-node')
            .classed('selected', false)
            .style('stroke', '#fff');

        this.svg.selectAll('.network-node')
            .filter(d => d.id === node.id)
            .classed('selected', true)
            .style('stroke', '#ffc107');

        // Update node details panel
        this.updateNodeDetails(node);
        this.selectedNode = node;
    }

    updateNodeDetails(node) {
        const detailsContainer = document.getElementById('nodeDetails');
        
        const eff = this.getEffectiveGnnProbability(node);
        const gnnUsed = eff.prob || 0;
        const gnnSource = eff.source || 'Unknown';

        const details = `
            <div class="row">
                <div class="col-md-6">
                    <h6>${node.id}</h6>
                    <p class="text-muted mb-2">${node.building_type || 'Unknown'}</p>
                    <p><strong>Degree:</strong> ${node.degree || 0}</p>
                    <p><strong>Capacity:</strong> ${node.max_capacity_kwh || 0} kWh</p>
                    <p><strong>Solar:</strong> ${node.installed_solar_kw || 0} kW</p>
                </div>
                <div class="col-md-6">
                    <p><strong>Battery Backup:</strong> ${node.has_battery_backup ? 'Yes' : 'No'}</p>
                    <p><strong>Status:</strong> ${node.status || 'Unknown'}</p>
                    <p><strong>Old Risk:</strong> ${(node.current_risk_score || 0).toFixed(3)}</p>
                    <p><strong>GNN Risk (${gnnSource}):</strong> ${gnnUsed.toFixed(3)}</p>
                    <button class="btn btn-primary btn-sm mt-2" onclick="predictNodeFromNetwork('${node.id}')">
                        <i class="fas fa-chart-line me-1"></i>Predict Failure
                    </button>
                </div>
            </div>
        `;
        
        detailsContainer.innerHTML = details;
    }

    showNodeTooltip(event, node) {
        // Create tooltip if it doesn't exist
        let tooltip = d3.select('body').select('.network-tooltip');
        if (tooltip.empty()) {
            tooltip = d3.select('body').append('div')
                .attr('class', 'network-tooltip')
                .style('position', 'absolute')
                .style('padding', '8px')
                .style('background', 'rgba(0, 0, 0, 0.8)')
                .style('color', 'white')
                .style('border-radius', '4px')
                .style('font-size', '12px')
                .style('pointer-events', 'none')
                .style('z-index', '1000');
        }

        tooltip.style('opacity', 1)
            .html(`<strong>${node.id}</strong><br/>${node.building_type || 'Unknown'}<br/>Degree: ${node.degree || 0}`)
            .style('left', (event.pageX + 10) + 'px')
            .style('top', (event.pageY - 10) + 'px');
    }

    hideNodeTooltip() {
        d3.select('.network-tooltip').style('opacity', 0);
    }

    updateLayout() {
        const layout = document.getElementById('layoutSelect').value;
        
        if (!this.simulation) return;

        switch (layout) {
            case 'circular':
                this.applyCircularLayout();
                break;
            case 'hierarchical':
                this.applyHierarchicalLayout();
                break;
            default:
                this.applyForceLayout();
                break;
        }
    }

    applyCircularLayout() {
        const radius = Math.min(this.width, this.height) / 2 - 50;
        const angleStep = (2 * Math.PI) / this.nodes.length;
        
        this.nodes.forEach((node, i) => {
            const angle = i * angleStep;
            node.fx = this.width / 2 + radius * Math.cos(angle);
            node.fy = this.height / 2 + radius * Math.sin(angle);
        });
        
        this.simulation.alpha(1).restart();
    }

    applyHierarchicalLayout() {
        // Simple hierarchical layout by building type
        const buildingTypes = [...new Set(this.nodes.map(n => n.building_type))];
        const levelHeight = this.height / (buildingTypes.length + 1);
        
        buildingTypes.forEach((type, typeIndex) => {
            const nodesOfType = this.nodes.filter(n => n.building_type === type);
            const levelY = (typeIndex + 1) * levelHeight;
            const nodeSpacing = this.width / (nodesOfType.length + 1);
            
            nodesOfType.forEach((node, nodeIndex) => {
                node.fx = (nodeIndex + 1) * nodeSpacing;
                node.fy = levelY;
            });
        });
        
        this.simulation.alpha(1).restart();
    }

    applyForceLayout() {
        this.nodes.forEach(node => {
            node.fx = null;
            node.fy = null;
        });
        this.simulation.alpha(1).restart();
    }

    updateNodeColors() {
        this.svg.selectAll('.network-node')
            .style('fill', d => this.getNodeColor(d));
        this.updateLegend();
    }

    updateNodeSizes() {
        this.svg.selectAll('.network-node')
            .attr('r', d => this.getNodeRadius(d));
    }

    updateLegend() {
        const legendContainer = document.getElementById('networkLegend');
        const colorBy = document.getElementById('colorBy').value;
        
        let legendHTML = '';
        
        switch (colorBy) {
            case 'building_type':
                const buildingTypes = [...new Set(this.nodes.map(n => n.building_type))];
                legendHTML = buildingTypes.map(type => `
                    <div class="d-flex align-items-center mb-2">
                        <div class="me-2" style="width: 16px; height: 16px; border-radius: 50%; background-color: ${this.colorScale(type)};"></div>
                        <small>${type}</small>
                    </div>
                `).join('');
                break;
            case 'risk_level':
                legendHTML = `
                    <div class="d-flex align-items-center mb-2">
                        <div class="me-2" style="width: 16px; height: 16px; border-radius: 50%; background-color: #198754;"></div>
                        <small>Low Risk</small>
                    </div>
                    <div class="d-flex align-items-center mb-2">
                        <div class="me-2" style="width: 16px; height: 16px; border-radius: 50%; background-color: #ffc107;"></div>
                        <small>Medium Risk</small>
                    </div>
                    <div class="d-flex align-items-center mb-2">
                        <div class="me-2" style="width: 16px; height: 16px; border-radius: 50%; background-color: #dc3545;"></div>
                        <small>High Risk</small>
                    </div>
                `;
                break;
            case 'gnn_risk':
                legendHTML = `
                    <div class="d-flex align-items-center mb-2">
                        <div class="me-2" style="width: 16px; height: 16px; border-radius: 50%; background-color: #198754;"></div>
                        <small>Low Risk</small>
                    </div>
                    <div class="d-flex align-items-center mb-2">
                        <div class="me-2" style="width: 16px; height: 16px; border-radius: 50%; background-color: #ffc107;"></div>
                        <small>Medium Risk</small>
                    </div>
                    <div class="d-flex align-items-center mb-2">
                        <div class="me-2" style="width: 16px; height: 16px; border-radius: 50%; background-color: #dc3545;"></div>
                        <small>High Risk</small>
                    </div>
                `;
                break;
            case 'centrality':
                legendHTML = `
                    <div class="d-flex align-items-center mb-2">
                        <div class="me-2" style="width: 16px; height: 16px; border-radius: 50%; background-color: #0066ff;"></div>
                        <small>Low Centrality</small>
                    </div>
                    <div class="d-flex align-items-center mb-2">
                        <div class="me-2" style="width: 16px; height: 16px; border-radius: 50%; background-color: #ff0066;"></div>
                        <small>High Centrality</small>
                    </div>
                `;
                break;
            case 'capacity':
                legendHTML = `
                    <div class="d-flex align-items-center mb-2">
                        <div class="me-2" style="width: 16px; height: 16px; border-radius: 50%; background-color: #17a2b8;"></div>
                        <small>Low Capacity</small>
                    </div>
                    <div class="d-flex align-items-center mb-2">
                        <div class="me-2" style="width: 16px; height: 16px; border-radius: 50%; background-color: #fd7e14;"></div>
                        <small>High Capacity</small>
                    </div>
                `;
                break;
        }
        
        legendContainer.innerHTML = legendHTML;
    }

    updateNetworkStats(stats) {
        if (stats) {
            document.getElementById('nodeCount').textContent = stats.node_count || 0;
            document.getElementById('edgeCount').textContent = stats.edge_count || 0;
            document.getElementById('networkDensity').textContent = (stats.density || 0).toFixed(3);
        }
    }

    showError(message) {
        const container = document.getElementById('networkContainer');
        container.innerHTML = `
            <div class="d-flex align-items-center justify-content-center h-100">
                <div class="text-center text-muted">
                    <i class="fas fa-exclamation-triangle fa-3x mb-3"></i>
                    <p>${message}</p>
                    <button class="btn btn-outline-primary" onclick="refreshNetwork()">
                        <i class="fas fa-sync-alt me-1"></i>Retry
                    </button>
                </div>
            </div>
        `;
    }

    resetZoom() {
        if (this.svg) {
            this.svg.transition().duration(750).call(
                d3.zoom().transform,
                d3.zoomIdentity
            );
        }
    }

    async refreshNetwork() {
        await this.loadNetworkData();
    }

    exportNetwork() {
        try {
            const svgElement = document.getElementById('networkSvg');
            const serializer = new XMLSerializer();
            const svgString = serializer.serializeToString(svgElement);
            
            const blob = new Blob([svgString], { type: 'image/svg+xml' });
            const url = URL.createObjectURL(blob);
            
            const link = document.createElement('a');
            link.href = url;
            link.download = 'power_grid_network.svg';
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            
            URL.revokeObjectURL(url);
            showNotification('Network exported successfully', 'success');
        } catch (error) {
            showNotification('Failed to export network', 'danger');
        }
    }
}

// Global functions
let networkViz = null;

function initializeNetwork() {
    networkViz = new NetworkVisualization();
    networkViz.init();
}

function updateLayout() {
    if (networkViz) {
        networkViz.updateLayout();
    }
}

function updateNodeColors() {
    if (networkViz) {
        networkViz.updateNodeColors();
    }
}

function updateNodeSizes() {
    if (networkViz) {
        networkViz.updateNodeSizes();
    }
}

function resetZoom() {
    if (networkViz) {
        networkViz.resetZoom();
    }
}

function refreshNetwork() {
    if (networkViz) {
        networkViz.refreshNetwork();
    }
}

function exportNetwork() {
    if (networkViz) {
        networkViz.exportNetwork();
    }
}

async function predictNodeFromNetwork(nodeId) {
    try {
        const response = await fetch(`/api/predict-node/${nodeId}`);
        const data = await response.json();
        if (data.success) {
            showNotification(`Prediction for ${nodeId}: ${data.prediction}`, 'success');
        } else {
            showNotification(`Prediction failed: ${data.error}`, 'danger');
        }
    } catch (error) {
        showNotification(`Prediction failed: ${error.message}`, 'danger');
    }
}

// Tooltip helper
function showNotification(message, type='info') {
    const alertContainer = document.getElementById('alertContainer');
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    alertContainer.appendChild(alertDiv);
    setTimeout(() => {
        alertDiv.classList.remove('show');
        alertDiv.classList.add('hide');
        alertDiv.remove();
    }, 4000);
}

// Initialize network visualization on page load
document.addEventListener('DOMContentLoaded', () => {
    initializeNetwork();

    // Add event listeners for controls
    const layoutSelect = document.getElementById('layoutSelect');
    const colorBy = document.getElementById('colorBy');
    const nodeSize = document.getElementById('nodeSize');
    const resetZoomBtn = document.getElementById('resetZoom');
    const refreshBtn = document.getElementById('refreshNetwork');
    const exportBtn = document.getElementById('exportSVG');

    if (layoutSelect) {
        layoutSelect.addEventListener('change', updateLayout);
    }

    if (colorBy) {
        colorBy.addEventListener('change', updateNodeColors);
    }

    if (nodeSize) {
        nodeSize.addEventListener('change', updateNodeSizes);
    }

    if (resetZoomBtn) {
        resetZoomBtn.addEventListener('click', resetZoom);
    }

    if (refreshBtn) {
        refreshBtn.addEventListener('click', refreshNetwork);
    }

    if (exportBtn) {
        exportBtn.addEventListener('click', exportNetwork);
    }
});
