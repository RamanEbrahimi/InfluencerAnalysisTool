# Influencer Multiplex Network Analysis Dashboard

An interactive Python dashboard for analyzing social media influencers across multiple platforms using **multiplex network analysis**. This project identifies hidden influencers, computes cross-platform importance metrics, and provides scientific insights into influencer behavior patterns.

## Overview

This dashboard analyzes influencers across **Instagram, Threads, TikTok, and YouTube** using advanced network science techniques:

- **Multiplex Network Analysis**: Each platform is a separate layer with inter-layer connections
- **Identity Resolution**: Fuzzy matching to link the same influencer across platforms
- **Shapley Coalition Impact**: Game-theoretic analysis of cross-platform influence
- **Centrality Metrics**: Degree, betweenness, eigenvector, and closeness centralities per layer
- **3D Visualization**: Layered platform representation with interactive exploration

## Architecture

### Core Components

```
src/
├── data_loader.py      # CSV standardization and data loading
├── identity.py         # Cross-platform identity resolution
├── graphs.py          # Per-layer and multiplex graph construction
├── metrics.py         # Centralities, Shapley values, and KPIs
└── app.py             # Dash web application
```

### Data Flow

1. **Data Loading** (`data_loader.py`): Standardizes platform CSVs with consistent schema
2. **Identity Resolution** (`identity.py`): Links influencers across platforms using fuzzy matching
3. **Graph Construction** (`graphs.py`): Builds similarity graphs per layer and multiplex supra-graph
4. **Metrics Computation** (`metrics.py`): Calculates centralities and Shapley coalition values
5. **Visualization** (`app.py`): Interactive Dash dashboard with multiple chart types

## Features

### 1. Multiplex Network Analysis
- **Per-layer graphs**: Topic similarity and numeric KNN graphs for each platform
- **Inter-layer coupling**: Entity-based connections across platforms
- **Supra-graph PageRank**: MultiRank algorithm for cross-platform importance

### 2. Advanced Metrics
- **Centrality Analysis**: Degree, betweenness, eigenvector, and closeness per layer
- **Shapley Coalition Impact**: Game-theoretic analysis of platform combinations
- **Hidden Influencers**: Identifies nodes strong across platforms but weak individually

### 3. Interactive Visualizations
- **3D Layered View**: Each platform as a separate z-plane
- **Node Statistics**: Detailed per-layer metrics with progress bars
- **Top 5 Charts**: Configurable by followers, engagement, reach, or multiplex rank
- **Engagement vs Followers**: Bubble chart with reach as bubble size

### 4. Scientific Insights
- **Cross-platform presence**: Nodes active in all selected layers
- **Coalition analysis**: Best platform combinations for each influencer
- **Multiplex reach**: Aggregated potential reach across platforms

## Screenshots

### Dashboard Overview
![Dashboard Overview](Screenshot%202025-09-18%20at%2015.38.19.png)
*Main dashboard showing KPI cards, layer selection, and multiplex graph visualization*

### Node Statistics Panel
![Node Statistics](Screenshot%202025-09-18%20at%2015.38.58.png)
*Detailed node statistics with per-layer metrics, centralities, and progress bars*

### 3D Multiplex Visualization
![3D Visualization](Screenshot%202025-09-18%20at%2015.39.27.png)
*3D layered view showing influencers across platforms with engagement vs followers*

## Installation & Setup

### Prerequisites
- Python 3.9+
- pip package manager

### Quick Start

```bash
# Clone or download the project
cd Influencer-Analysis-Dashboard

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
python -m src.app
```

The dashboard will be available at `http://127.0.0.1:8050/`

## Usage Guide

### 1. Layer Selection
- Select platforms to analyze (Instagram, Threads, TikTok, YouTube)
- Choose graph type: Topic similarity or Numeric KNN
- Optionally highlight a specific layer

### 2. Node Analysis
- Click any node in the graph to see detailed statistics
- View per-layer metrics: followers, engagement, reach, topic
- Examine centrality measures with progress bars and numeric values

### 3. Shapley Coalition Analysis
- Select base metric (followers, engagement, reach)
- Adjust coalition size (1-4 platforms)
- Identify influencers with highest impact in specific platform combinations

### 4. Cross-Platform Insights
- View nodes present in all selected layers
- Analyze multiplex reach and importance scores
- Explore 3D visualization for spatial understanding

## Scientific Methodology

### Multiplex Network Construction
1. **Intra-layer edges**: Topic similarity (TF-IDF cosine) and numeric KNN
2. **Inter-layer edges**: Entity-based coupling with configurable weights
3. **Supra-graph**: Combined representation for multiplex analysis

### Identity Resolution
- **Fuzzy matching**: RapidFuzz for name and username similarity
- **Entity grouping**: Connected components across exact and fuzzy matches
- **Canonical representation**: Highest-follower account as primary identity

### Shapley Value Computation
- **Value function**: v(S) = log(1 + Σ normalized_metric)
- **Coalition analysis**: Best platform combinations per influencer
- **Fair attribution**: Game-theoretic approach to cross-platform impact

## Data Schema

### Input Data
Each platform CSV should contain:
- `rank`: Platform-specific ranking
- `name`: Influencer display name
- `username`: Platform username
- `followers`: Follower count
- `engagement_rate`: Engagement percentage
- `country`: Geographic location
- `topic_of_influence`: Content categories
- `potential_reach`: Estimated reach

### Output Metrics
- **Centralities**: Degree, betweenness, eigenvector, closeness
- **Multiplex scores**: Supra-graph PageRank aggregation
- **Coalition values**: Shapley-based platform combination impact
- **Cross-platform presence**: Layer activity counts

## Customization

### Adding New Platforms
1. Add CSV to `data/` directory
2. Update `DATA_FILES` in `data_loader.py`
3. Add platform color to `LAYER_COLORS` in `app.py`

### Modifying Metrics
- Edit `compute_layer_centralities()` for new centrality measures
- Update `compute_shapley_coalitions()` for different value functions
- Customize `summarize_kpis()` for additional summary statistics

### UI Themes
- Change `dbc.themes.COSMO` to other Bootstrap themes
- Modify `LAYER_COLORS` for platform-specific styling
- Adjust chart templates in Plotly calls

## Technical Details

### Dependencies
- **Dash**: Web application framework
- **NetworkX**: Graph analysis and algorithms
- **Plotly**: Interactive visualizations
- **scikit-learn**: Machine learning utilities
- **RapidFuzz**: Fuzzy string matching
- **pandas/numpy**: Data manipulation

### Performance Considerations
- **Graph size**: Optimized for datasets with <10K nodes per layer
- **Shapley computation**: Exponential complexity; limit to 4 platforms
- **Real-time updates**: Callbacks optimized for responsive interaction

## References

- **Multiplex Networks**: Boccaletti et al. (2014) "The structure and dynamics of multilayer networks"
- **Shapley Values**: Shapley (1953) "A value for n-person games"
- **Network Centrality**: Freeman (1978) "Centrality in social networks"

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Built with love using Python, Dash, and NetworkX**