using System;
using System.Configuration;
using System.Globalization;
using System.Threading;
using Ardalis.GuardClauses;
using Microsoft.Azure.Functions.Worker;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Logging;
using NexusLocate.Application.Interfaces;
using NexusLocate.Application.Location.Dto;
using NexusLocate.Application.Scan.Dto;
using NexusLocate.Application.Shared;
using NexusLocate.Application.Utilities;
using NexusLocate.ConsumerFunc.Services;
using NexusLocate.ConsumerFunc.Services.Cache;
using NexusLocate.Domain.Entities;
using NexusLocate.Infrastructure.Repositories;
using YamlDotNet.Core.Tokens;
using static System.Runtime.InteropServices.JavaScript.JSType;
using static Microsoft.ApplicationInsights.MetricDimensionNames.TelemetryContext;

namespace ScanDataProcessor;

public class SuperAmazingAlgo
{
    private readonly ILogger<SuperAmazingAlgo> _logger;
    private readonly IScanRepository _scanRepository;
    private readonly ISiteRepository _siteRepository;
    private readonly INexusLocateService _nexusLocateService;
    private readonly IUnitOfWork _unitOfWork;
    private readonly IConfigurationSettingRepository _configurationSettingRepository;
    private readonly ILocationCoordinateRepository _locationCoordinateRepository;
    private readonly ILocationRepository _locationRepository;
    private readonly IObservationRepository _observationRepository;
    private readonly IUpdatedLocationRepository _updatedLocationRepository;

    private readonly HashSet<string> _processedRfids = new HashSet<string>();

    public SuperAmazingAlgo(ILoggerFactory loggerFactory, IScanRepository scanRepository,
        ISiteRepository siteRepository, INexusLocateService nexusLocateService,
        IUnitOfWork unitOfWork, IConfigurationSettingRepository configurationSettingRepository,
        ILocationCoordinateRepository locationCoordinateRepository,
        ILocationRepository locationRepository, IObservationRepository observationRepository,
        IUpdatedLocationRepository updatedLocationRepository)
    {
        _logger = loggerFactory.CreateLogger<SuperAmazingAlgo>();
        _scanRepository = scanRepository ?? throw new ArgumentNullException(nameof(scanRepository));
        _siteRepository = siteRepository ?? throw new ArgumentNullException(nameof(siteRepository));
        _nexusLocateService = nexusLocateService ?? throw new ArgumentNullException(nameof(nexusLocateService));
        _unitOfWork = unitOfWork ?? throw new ArgumentNullException(nameof(unitOfWork));
        _configurationSettingRepository = configurationSettingRepository ?? throw new ArgumentNullException(nameof(configurationSettingRepository));
        _locationCoordinateRepository = locationCoordinateRepository ?? throw new ArgumentNullException(nameof(locationCoordinateRepository));
        _locationRepository = locationRepository ?? throw new ArgumentNullException(nameof(locationRepository));
        _observationRepository = observationRepository ?? throw new ArgumentNullException(nameof(observationRepository));
        _updatedLocationRepository = updatedLocationRepository ?? throw new ArgumentNullException(nameof(updatedLocationRepository));
    }

    // This function runs every 5 seconds
    [Function("SuperAmazingAlgo")]
    public async Task Run([TimerTrigger("*/5 * * * * *")] TimerInfo myTimer, CancellationToken cancellationToken)
    {
        _logger.LogInformation("Function executed at: {time}", DateTime.Now);

        int limit = 50; // Replace with desired limit

        var sites = await _siteRepository.FindAsync(
        t => !t.IsDeleted, cancellationToken);

        try
        {
            foreach (var site in sites)
            {
               
                var capabilityFlags = await _nexusLocateService.GetCapabilityFlags(site.Id, cancellationToken);
                var allowNexusLocateCapabilityEnabled = capabilityFlags?.AllowNexusLocate ?? false;
                
                if (allowNexusLocateCapabilityEnabled)
                {
                    var configurationSettings = await ConfigurationSetting(site.Id);
                    if (configurationSettings == null)
                    {
                        _logger.LogInformation("Nexus Locate configuration not found");
                        continue;
                    }
                    //Fetch parking spots (cached per site)
                    var parkingSpots = await ParkingSpotCache.GetOrUpdateAsync(
                        site.Id,
                        async (id) => await GetParkingSpotsAsync(id, cancellationToken)
                    );
                    await DeleteObservationsAsync(site.Id, configurationSettings.ObservationCutoffMinutes);
                    var unprocessedScans = await GetLatestUnprocessedScans(site.Id, limit, cancellationToken);

                    foreach (var scan in unprocessedScans)
                    {
                        await ProcessIndividualScanAsync(scan, site.Id, configurationSettings, parkingSpots, cancellationToken);
                    }
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error while fetching scan or config data");
        }

        if (myTimer.ScheduleStatus is not null)
        {
            _logger.LogInformation("Next schedule at: {next}", myTimer.ScheduleStatus.Next);
        }
    }
    private async Task ProcessIndividualScanAsync(Scan scan,Guid siteId,ConfigurationSetting configurationSettings,List<ParkingSpotDto> parkingSpots,CancellationToken cancellationToken = default)
    {
        try
        {
            if (!IsValidRfidTag(scan.TagName))
            {
                await UpdateScanAsProcessedAsync(scan.Id, "Ignore Inventory - Invalid Tag", cancellationToken);
                return;
            }

            if (!IsValidSpeed(scan.Speed, configurationSettings.MinSpeed, configurationSettings.MaxSpeed))
            {
                await UpdateScanAsProcessedAsync(scan.Id, "Ignore Inventory - Speed is not valid", cancellationToken);
                return;
            }

            if (_processedRfids.Contains(scan.TagName))
            {
                await UpdateScanAsProcessedAsync(scan.Id, "Ignore Inventory - Already Moved", cancellationToken);
                return;
            }

            if (IsDriverInParkingSpot(scan, siteId))
            {
                await UpdateScanAsProcessedAsync(scan.Id, "Ignore Inventory - Driver inside the parking spot", cancellationToken);
                return;
            }

            // Calculate valid parking spots with distance
            var distanceToParkingSpot = new SortedDictionary<double, ParkingSpotDto>();

            foreach (var spot in parkingSpots)
            {
                bool isAngleValid = true;

                if (!string.IsNullOrEmpty(scan.Barrier) && spot.X.HasValue && spot.Y.HasValue)
                {
                    isAngleValid = GeofenceUtils.IsValidateAngle(
                        "1",
                        Convert.ToDouble(scan.Barrier, CultureInfo.InvariantCulture),
                        scan.Latitude,
                        scan.Longitude,
                        spot.X.Value,
                        spot.Y.Value,
                        configurationSettings);
                }

                if (!isAngleValid) continue;

                double distance = GeofenceUtils.DistanceInFeet(
                    scan.Latitude,
                    scan.Longitude,
                    spot.X.Value,
                    spot.Y.Value);

                if (distance < configurationSettings.MaxSlotDistance && !distanceToParkingSpot.ContainsKey(distance))
                {
                    distanceToParkingSpot.Add(distance, spot);
                }
            }

            var closestParkingSpots = GeofenceUtils.GetClosestParkingSpots(distanceToParkingSpot, configurationSettings.MaxSlots);

            if (closestParkingSpots.Count == 0)
            {
                await UpdateScanAsProcessedAsync(scan.Id, "Ignore Inventory - Unable to pick the location", cancellationToken);
                _logger.LogDebug("No valid parking spots found for RFID: {RfidTag}", scan.TagName);
                return;
            }

            var nearestParkingSpot = closestParkingSpots[0];
            _logger.LogDebug("Nearest parking spot: {ParkingSpotName} for RFID: {RfidTag}", nearestParkingSpot.Name, scan.TagName);

            var observation = Observation.Create(
                id: Guid.NewGuid().ToString(),
                siteId: siteId.ToString(),
                deviceId: scan.DeviceId,
                rfid: scan.TagName,
                latitude: scan.Latitude,
                longitude: scan.Longitude,
                heading: 0,
                speed: scan.Speed,
                updatedLocation: nearestParkingSpot.Name,
                updatedLocationId: nearestParkingSpot.Id.ToString(),
                version: 1
            );

            await _observationRepository.AddAsync(observation, cancellationToken);
            await _unitOfWork.SaveChangesAsync(cancellationToken);

            var activeObservations = await GetActiveObservationsAsync(scan.TagName, siteId.ToString());

            if (activeObservations.Count >= configurationSettings.ObservationsCount)
            {
                await UpdateRfidLocationAsync(scan.Id, siteId, Guid.NewGuid(), nearestParkingSpot.Id.ToString(), cancellationToken);

                _ = UpdateScanWithLocationSuccessAsync(scan.Id, "Location is Updated: " + nearestParkingSpot.Name, cancellationToken);

                _processedRfids.Add(scan.TagName);

                _logger.LogInformation("Location updated successfully for RFID: {RfidTag} to {LocationName}",
                    scan.TagName, nearestParkingSpot.Name);
            }
            else
            {
                _logger.LogDebug("Insufficient observations ({Count}/{Required}) for RFID: {RfidTag}",
                    activeObservations.Count, configurationSettings.ObservationsCount, scan.TagName);
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error processing scan {ScanId}", scan.Id);
            throw;
        }
    }

    public async Task DeleteObservationsAsync(Guid siteId, int observationCutoffMinutes)
    {
        if (observationCutoffMinutes <= 0) return;

        var cutoffTime = DateTime.UtcNow.AddMinutes(-observationCutoffMinutes);

        var staleObservations = await _observationRepository.FindAsync(
            t => t.SiteId == siteId.ToString() && t.CreatedOn < cutoffTime
        );

        if (staleObservations?.Any() != true) return;

        _observationRepository.DeleteMany(staleObservations); // Use batch delete method if available
        await _unitOfWork.SaveChangesAsync();

    }

    private async Task UpdateScanWithLocationSuccessAsync(Guid scanId, string comment, CancellationToken cancellationToken = default)
    {
        var scan = await _scanRepository.FindOneAsync(t => t.Id == scanId, cancellationToken);

        if (scan != null)
        {
            scan.IsProcess = true;
            scan.Comment = comment;
            //scan.TagName = tagName;
            scan.UpdatedOn = DateTimeOffset.UtcNow;

            await _unitOfWork.SaveChangesAsync(cancellationToken);
        }
    }

    private async Task UpdateRfidLocationAsync(
       Guid tagName,
       Guid siteId,
       Guid deviceId,
       string nearestParkingSpotId,
       CancellationToken cancellationToken = default)
    {
        var updatedLocationId = Guid.NewGuid();

        var updatedLocation = UpdatedLocation.Create(
            updatedLocationId,
            siteId,
            deviceId,
            tagName,
            Guid.Parse(nearestParkingSpotId),  // Convert location ID string to Guid
            isShared: true                     // or false, depending on your logic
        );

        await _updatedLocationRepository.AddAsync(updatedLocation, cancellationToken);
        await _unitOfWork.SaveChangesAsync(cancellationToken);
    }

    public async Task<List<Observation>> GetActiveObservationsAsync(string rfidTag, string siteId)
    {
        var observations = await _observationRepository.FindAsync(
            t => t.SiteId == siteId && t.RFID == rfidTag);

        return observations.ToList();
    }

    public async Task<List<ConfigurationSetting>> GetObservationsThresholdAsync(Guid siteId)
    {
        var ConfigurationSettings = await _configurationSettingRepository.FindAsync(
            t => t.SiteId == siteId);

        return ConfigurationSettings.ToList();
    }

    public async Task<ConfigurationSetting> ConfigurationSetting(Guid siteId)
    {
        var ConfigurationSettings = await _configurationSettingRepository.FindAsync(
            t => t.SiteId == siteId);

        return ConfigurationSettings.FirstOrDefault();
    }
    private async Task<List<ParkingSpotDto>> GetParkingSpotsAsync(Guid siteId, CancellationToken cancellationToken)
    {
        var locations = await _locationRepository.FindAsync(t => t.SiteId == siteId, cancellationToken);
        var locationIds = locations.Select(l => l.Id).ToList();

        var coordinates = await _locationCoordinateRepository.FindAsync(
            c => !c.IsDeleted && locationIds.Contains(c.LocationId),
            cancellationToken);

        var latestCoordinates = coordinates
            .GroupBy(c => c.LocationId)
            .Select(g => g.OrderByDescending(c => c.Id).First())
            .ToDictionary(c => c.LocationId);

        var parkingSpots = new List<ParkingSpotDto>();

        foreach (var location in locations)
        {
            if (!latestCoordinates.TryGetValue(location.Id, out var coord))
                continue;

            parkingSpots.Add(new ParkingSpotDto
            {
                Id = location.Id,
                Name = location.Name,
                X = coord.Latitude,
                Y = coord.Longitude,
                SiteId = location.SiteId,
                CreatedOn = location.CreatedOn,
                CreatedBy = location.CreatedBy
            });
        }

        return parkingSpots;
    }
    private async Task<SortedDictionary<double, ParkingSpotDto>> GetAllValidParkingSpotsAsync(Scan scan, bool angleRequire,
        Guid siteId, int maxSlotDistance,
        CancellationToken cancellationToken = default)
    {
        // Initialize result dictionary
        var distanceToParkingSpot = new SortedDictionary<double, ParkingSpotDto>();

        // Fetch locations for the site
        var locations = await _locationRepository.FindAsync(
            t => t.SiteId == siteId,
            cancellationToken);

        // Fetch configuration settings for the site
        var configurationSetting = await _configurationSettingRepository.FindAsync(
            t => t.SiteId == siteId,
            cancellationToken);

        // Assuming 'locationCoordinates' is available in the context
        // Map location IDs to coordinates
        int limit = 1;
        var locationCoordinates = await GetLocationCoordinatesBySiteAsync(siteId, cancellationToken);
        var coordinateMap = locationCoordinates.ToDictionary(lc => lc.LocationId, lc => lc);

        // Assuming 'observation' contains the relevant GPS data (previously named 'inventory')
        foreach (var location in locations)
        {
            if (!coordinateMap.TryGetValue(location.Id, out var coordinates))
                continue;

            var parkingSpot = new ParkingSpotDto
            {
                Id = location.Id,
                Name = location.Name,
                X = coordinates.Latitude,
                Y = coordinates.Longitude,
                SiteId = location.SiteId,
                // Geofence = location.Geofence,
                //IsOccupied = location.IsOccupied,
                //  IsShared = location.IsShared,
                CreatedOn = location.CreatedOn,
                CreatedBy = location.CreatedBy
            };

            bool angle = true;

            if (angleRequire)
            {
                // Validate GPS/heading data before calculating angle
                if (scan.Barrier == null || parkingSpot.X == null || parkingSpot.Y == null)
                {
                    angle = false;
                }
                else
                {
                    const string RightOrientation = "1";
                    angle = GeofenceUtils.IsValidateAngle(
                        RightOrientation,
                        Convert.ToDouble(scan.Barrier, CultureInfo.InvariantCulture),
                        scan.Latitude,
                        scan.Longitude,
                        parkingSpot.X.Value,
                        parkingSpot.Y.Value,
                        configurationSetting.First());
                }
            }

            if (!angle)
                continue;

            // Calculate distance between observation and parking spot
            double distance = GeofenceUtils.DistanceInFeet(
                scan.Latitude,
                scan.Longitude,
                parkingSpot.X.Value,
                parkingSpot.Y.Value);

            if (distance < maxSlotDistance)
            {
                if (!distanceToParkingSpot.ContainsKey(distance))
                {
                    distanceToParkingSpot.Add(distance, parkingSpot);
                }
            }
        }

        return distanceToParkingSpot;
    }

    /// <summary>
    /// Fetches configuration values (speed limits, slot limits, etc.) for a given site.
    /// </summary>
    private async Task<(double MinSpeed, double MaxSpeed, int MaxSlots, int MaxSlotDistance)> GetConfigurationAsync(
        Guid siteId,
        CancellationToken cancellationToken = default)
    {
        var configurationSetting = (await _configurationSettingRepository.FindAsync(
            t => t.SiteId == siteId,
            cancellationToken
        )).FirstOrDefault();

        if (configurationSetting != null)
        {
            var minSpeed = configurationSetting.MinSpeed;
            var maxSpeed = configurationSetting.MaxSpeed;
            var maxSlots = configurationSetting.MaxSlots;
            var maxSlotDistance = configurationSetting.MaxSlotDistance;

            return (minSpeed, maxSpeed, maxSlots, maxSlotDistance);
        }

        _logger.LogWarning("No configuration found for site {SiteId}", siteId);
        return (0, 0, 0, 0); // Default values
    }

    public static bool IsDriverInParkingSpot(Scan inventory, Guid siteId)
    {
        // Placeholder logic
        return false;
    }

    private async Task UpdateScanAsProcessedAsync(Guid scanId, string comment, CancellationToken cancellationToken = default)
    {
        var scan = await _scanRepository.FindOneAsync(t => t.Id == scanId, cancellationToken);

        if (scan != null)
        {
            scan.IsProcess = true;
            scan.Comment = comment;
            scan.UpdatedOn = DateTimeOffset.UtcNow;

            await _unitOfWork.SaveChangesAsync(cancellationToken);
        }
    }

    private async Task<List<Scan>> GetLatestUnprocessedScans(Guid siteId, int limit, CancellationToken cancellationToken)
    {
        if (limit <= 0)
        {
            return new List<Scan>();
        }

        try
        {
            return await GetUnprocessedScansAsync(siteId, limit, cancellationToken);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error retrieving scans for site {SiteId}", siteId);
            return new List<Scan>();
        }
    }
    private async Task<List<LocationCoordinateDto>> GetLocationCoordinatesBySiteAsync(Guid siteId, CancellationToken cancellationToken)
    {

        // Fetch all non-deleted coordinates
        var coordinates = await _locationCoordinateRepository.FindAsync(c => !c.IsDeleted, cancellationToken);

        // Group by LocationId and pick the latest coordinate for each
        var latestCoordinates = coordinates
            .GroupBy(c => c.LocationId)
            .Select(g => g.OrderByDescending(c => c.Id).First())
            .Select(c => new LocationCoordinateDto
            {
                LocationId = c.LocationId,
                Latitude = c.Latitude,
                Longitude = c.Longitude
            })
            .ToList();

        return latestCoordinates;
    }

    private async Task<List<Scan>> GetUnprocessedScansAsync(Guid siteId, int limit, CancellationToken cancellationToken)
    {
        // Step 1: Fetch all unprocessed scans for the site before cutoff time
        var cutoffTime = DateTime.UtcNow.AddMilliseconds(-10000); // 10 seconds ago
        var scans = (await _scanRepository.FindAsync(
            t => t.CreatedOn < cutoffTime &&
                 !t.IsDeleted &&
                 !t.IsProcess &&
                 t.SiteId == siteId,
            cancellationToken
        ))
        .OrderByDescending(t => t.CreatedOn)
        .ToList();

        // Step 2: Group by RFID tag and select latest per tag
        var uniqueLatestPerRfid = scans
            .GroupBy(t => t.TagName)
            .Select(g => g.First()) // Already ordered DESC
            .OrderBy(t => t.Id)     // Match SQL: final order by ID
            .Take(limit)               // Limit to top 50
            .ToList();

        // Step 3: Mark older duplicates as processed
        var duplicateRecords = scans
            .Where(scan => !uniqueLatestPerRfid.Any(unique => unique.Id == scan.Id))
            .ToList();

        foreach (var dup in duplicateRecords)
        {
            dup.IsProcess = true;
            dup.Comment = "Ignored duplicated RFID";
            _scanRepository.Update(dup);
        }

        if (duplicateRecords.Any())
            await _unitOfWork.SaveChangesAsync(cancellationToken);
        return uniqueLatestPerRfid;
       
    }

    private bool IsValidRfidTag(string rfidTag)
    {
        return !string.IsNullOrEmpty(rfidTag) && rfidTag.Length > 20;
    }

    private bool IsValidSpeed(double speed, double minSpeed, double maxSpeed)
    {
        return speed >= minSpeed && speed <= maxSpeed;
    }

    private bool IsValidSlot(double speed, int maxSlots, int maxSlotDistance)
    {
        return speed >= maxSlots && speed <= maxSlotDistance;
    }
}

/// <summary>
/// Checks if the driver (based on scan data) is in a parking spot for the given site.
/// </summary>
/// <param name="inventory">Scan data row object with position info.</param>
/// <param name="SiteId">ID of the site to check against.</param>
/// <returns>True if the driver is in a valid parking spot; otherwise, false.</returns>
//public static bool IsDriverInParkingSpot(ScanRecord inventory, String SiteId)
//{
//    double lat1 = inventory.Latitude;
//    double lng1 = inventory.Longitude;

//    // Fetch all parking spots for the site
//    List<Locations> allParkingSpots = Locations.GetBySite(site)
//        .OrderBy(p => p.Name)
//        .ToList();

//    foreach (var spot in allParkingSpots)
//    {
//        double[] latList = GeofenceUtils.GetLat(spot.Geofence);
//        double[] lngList = GeofenceUtils.GetLng(spot.Geofence);

//        // Assuming polygon has 4 points
//        Polygon2D poly = new Polygon2D(latList, lngList, 4);

//        double lat = Math.Abs(lat1);
//        double lng = Math.Abs(lng1);

//        if (poly.Contains(lat, lng))
//        {
//            return true; // Point is inside a polygon
//        }
//    }

//    return false; // Point not found in any polygon
//}

