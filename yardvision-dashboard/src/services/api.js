// API service for fetching data from backend

const API_BASE = '/api'

export const fetchDashboardData = async (date = null) => {
  try {
    const url = date 
      ? `${API_BASE}/dashboard/data?date=${date}`
      : `${API_BASE}/dashboard/data`
    const response = await fetch(url)
    if (!response.ok) {
      throw new Error('Failed to fetch dashboard data')
    }
    return await response.json()
  } catch (error) {
    console.error('Error fetching dashboard data:', error)
    return null
  }
}

export const fetchDashboardEvents = async (limit = 1000, date = null) => {
  try {
    let url = `${API_BASE}/dashboard/events?limit=${limit}`
    if (date) {
      url += `&date=${date}`
    }
    const response = await fetch(url)
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

export const fetchInventoryData = async () => {
  try {
    const response = await fetch(`${API_BASE}/inventory`)
    if (!response.ok) {
      throw new Error('Failed to fetch inventory data')
    }
    return await response.json()
  } catch (error) {
    console.error('Error fetching inventory data:', error)
    return null
  }
}

export const fetchCameras = async (forceRefresh = false) => {
  try {
    const url = forceRefresh 
      ? `${API_BASE}/cameras?force=true`
      : `${API_BASE}/cameras`
    const response = await fetch(url)
    if (!response.ok) {
      throw new Error('Failed to fetch cameras')
    }
    const data = await response.json()
    return data.cameras || []
  } catch (error) {
    console.error('Error fetching cameras:', error)
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


