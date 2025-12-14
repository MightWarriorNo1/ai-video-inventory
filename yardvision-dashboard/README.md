# YardVision AI Dashboard

A modern React dashboard for displaying YOLO and OCR processing results from the Trailer Vision Edge application.

## Features

- **Dashboard**: Overview with KPIs, charts, and camera health
- **Inventory**: Trailer inventory management and tracking
- **Yard View**: Visual yard layout with spot occupancy
- **Camera Health**: Real-time camera status and metrics
- **Events**: Event log with filtering capabilities
- **Configuration**: System configuration management
- **Reports**: Daily, weekly, and monthly reports

## Setup

### Prerequisites

- Node.js 16+ and npm/yarn

### Installation

1. Navigate to the dashboard directory:
```bash
cd yardvision-dashboard
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm run dev
```

4. Open your browser and navigate to:
```
http://localhost:3000
```

## Building for Production

```bash
npm run build
```

The built files will be in the `dist` directory.

## Integration with Backend

The dashboard is designed to work with the existing Flask metrics server running on port 8080. The Vite configuration includes a proxy to forward API requests to the backend.

To connect to real data, update the API endpoints in the components to fetch from:
- `/api/metrics.json` - Metrics data
- `/api/events` - Event data
- `/api/cameras` - Camera data

Currently, the dashboard displays mock data. To switch to real data, replace the mock data imports with API calls.

## Project Structure

```
yardvision-dashboard/
├── src/
│   ├── components/     # Reusable components
│   ├── pages/         # Page components
│   ├── data/          # Mock data
│   ├── App.jsx        # Main app with routing
│   └── main.jsx       # Entry point
├── index.html
├── package.json
└── vite.config.js
```

## Development

The dashboard uses:
- **React 18** for UI
- **React Router** for navigation
- **Recharts** for data visualization
- **Vite** for build tooling

## Notes

- This dashboard is separate from the existing dashboard in the `web/` directory
- The original dashboard remains unchanged and can still be used for video upload and processing
- This React dashboard is designed to display results from YOLO and OCR processing


