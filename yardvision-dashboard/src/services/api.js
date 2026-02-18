// API service for fetching data from backend (video_frames.db)

const API_BASE = '/api'

export const fetchDashboardData = async (date = null) => {
  const url = date
    ? `${API_BASE}/dashboard/data?date=${date}`
    : `${API_BASE}/dashboard/data`
  console.log('[api] fetchDashboardData', url)
  try {
    const response = await fetch(url)
    if (!response.ok) {
      throw new Error('Failed to fetch dashboard data')
    }
    const data = await response.json()
    console.log('[api] fetchDashboardData ok', data?.kpis ? 'kpis present' : 'no kpis')
    return data
  } catch (error) {
    console.error('[api] fetchDashboardData error', error)
    return null
  }
}

export const fetchDashboardEvents = async (limit = 1000, date = null) => {
  let url = `${API_BASE}/dashboard/events?limit=${limit}`
  if (date) {
    url += `&date=${date}`
  }
  console.log('[api] fetchDashboardEvents', url)
  try {
    const response = await fetch(url)
    if (!response.ok) {
      throw new Error('Failed to fetch events')
    }
    const data = await response.json()
    const events = data.events || []
    console.log('[api] fetchDashboardEvents ok count=', events.length)
    return events
  } catch (error) {
    console.error('[api] fetchDashboardEvents error', error)
    return []
  }
}

export const fetchInventoryData = async () => {
  const url = `${API_BASE}/inventory`
  console.log('[api] fetchInventoryData', url)
  try {
    const response = await fetch(url)
    if (!response.ok) {
      throw new Error('Failed to fetch inventory data')
    }
    const data = await response.json()
    console.log('[api] fetchInventoryData ok trailers=', data?.trailers?.length ?? 0)
    return data
  } catch (error) {
    console.error('[api] fetchInventoryData error', error)
    return null
  }
}

export const fetchCameras = async (forceRefresh = false) => {
  const url = forceRefresh
    ? `${API_BASE}/cameras?force=true`
    : `${API_BASE}/cameras`
  console.log('[api] fetchCameras', url)
  try {
    const response = await fetch(url)
    if (!response.ok) {
      throw new Error('Failed to fetch cameras')
    }
    const data = await response.json()
    const cameras = data.cameras || []
    console.log('[api] fetchCameras ok count=', cameras.length)
    return cameras
  } catch (error) {
    console.error('[api] fetchCameras error', error)
    return []
  }
}

export const fetchEvents = async (limit = 100) => {
  try {
    const response = await fetch(`/events?limit=${limit}`)
    if (!response.ok) {
      throw new Error('Failed to fetch events')
    }
    const data = await response.json()
    return data.events || []
  } catch (error) {
    console.error('Error fetching events:', error)
    return []
  }
}

export const fetchVideoFrameRecords = async (options = {}) => {
  const {
    limit = 50,
    offset = 0,
    is_processed = null,
    camera_id = null
  } = options

  let url = `${API_BASE}/video-frame-records?limit=${limit}&offset=${offset}`
  if (is_processed !== null) {
    url += `&is_processed=${is_processed}`
  }
  if (camera_id) {
    url += `&camera_id=${encodeURIComponent(camera_id)}`
  }
  console.log('[api] fetchVideoFrameRecords', url)
  try {
    const response = await fetch(url)
    if (!response.ok) {
      throw new Error('Failed to fetch video frame records')
    }
    const data = await response.json()
    console.log('[api] fetchVideoFrameRecords ok records=', data?.records?.length ?? 0, 'total=', data?.total ?? 0)
    return data
  } catch (error) {
    console.error('[api] fetchVideoFrameRecords error', error)
    return { records: [], stats: {}, total: 0 }
  }
}

export const fetchYardViewData = async () => {
  const url = `${API_BASE}/yard-view`
  console.log('[api] fetchYardViewData', url)
  try {
    const response = await fetch(url)
    if (!response.ok) {
      throw new Error('Failed to fetch yard view data')
    }
    const data = await response.json()
    console.log('[api] fetchYardViewData ok spots=', data?.spots?.length ?? 0, 'lanes=', data?.lanes?.length ?? 0)
    return data
  } catch (error) {
    console.error('[api] fetchYardViewData error', error)
    return { spots: [], lanes: [] }
  }
}

export const fetchReportsData = async () => {
  const url = `${API_BASE}/reports`
  console.log('[api] fetchReportsData', url)
  try {
    const response = await fetch(url)
    if (!response.ok) {
      throw new Error('Failed to fetch reports data')
    }
    const data = await response.json()
    console.log('[api] fetchReportsData ok')
    return data
  } catch (error) {
    console.error('[api] fetchReportsData error', error)
    return null
  }
}


