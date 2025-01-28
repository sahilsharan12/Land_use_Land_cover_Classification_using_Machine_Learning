// Step 1: Define Pune district and filter Sentinel-2 image

Map.addLayer(pune, {}, 'Pune');
Map.centerObject(pune, 12);
// Load Sentinel-2 data and preprocess
var filter_Sentinel_2 = Sentinel.filterBounds(pune)
    .filterMetadata('CLOUDY_PIXEL_PERCENTAGE', 'less_than', 5)
    .filterDate('2024-01-01', '2024-10-01')
    .mean() // Reduce to a single image
    .clip(pune);
    Map.addLayer(filter_Sentinel_2,imageVisParam2,'Sentinel')
// Band assignments
var NIR = 'B8';
var Red = 'B4';
var Green = 'B3';
var SWIR1 = 'B11';
var Blue = 'B2';
var Band5 = 'B5';

// Cloud mask function
var cloudMask = function(image) {
  var qa = image.select('QA60');
  var cloudBitMask = 1 << 10;
  var cirrusBitMask = 1 << 11;
  var mask = qa.bitwiseAnd(cloudBitMask).eq(0).and(qa.bitwiseAnd(cirrusBitMask).eq(0));
  return image.updateMask(mask).divide(10000); // Scale reflectance to [0,1]
};

// Add indices function
var addIndices = function(image) {
  var ndvi = image.normalizedDifference([NIR, Red]).rename('NDVI');
  var mndwi = image.normalizedDifference([Green, SWIR1]).rename('MNDWI');
  var ndbi = image.normalizedDifference([SWIR1, NIR]).rename('NDBI');
  return image.addBands(ndvi).addBands(mndwi).addBands(ndbi);
};

// Process dataset
var dataset = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .filterBounds(pune)
    .filterDate('2024-01-01', '2024-10-01');
    // Get the count of images in the filtered collection
var imageCount = dataset.size();

// Print the number of images
print('Number of Sentinel-2 images in the specified range: ', imageCount);
var datasetNoCloud = dataset.map(cloudMask).map(addIndices);
var datasetNoCloud_Median = datasetNoCloud.mean();

// Compute min/max indices
var MaxNDVI = datasetNoCloud.select('NDVI').max().rename('MaxNDVI');
var MaxMNDWI = datasetNoCloud.select('MNDWI').max().rename('MaxMNDWI');
var MinNDVI = datasetNoCloud.select('NDVI').min().rename('MinNDVI');
var MinMNDWI = datasetNoCloud.select('MNDWI').min().rename('MinMNDWI');
var MinNDBI = datasetNoCloud.select('NDBI').min().rename('MinNDBI');
var MaxNDBI = datasetNoCloud.select('NDBI').max().rename('MaxNDBI');
// Add layers to the map
// Map.addLayer(MaxNDVI.clip(pune), {}, 'MaxNDVI');
// Map.addLayer(MaxMNDWI.clip(pune), {}, 'MaxMNDWI');
// Map.addLayer(MinNDVI.clip(pune), {}, 'MinNDVI');
// Map.addLayer(MinMNDWI.clip(pune), {}, 'MinMNDWI');
// Map.addLayer(MinNDBI.clip(pune), {}, 'MinNDBI');
// Map.addLayer(MaxNDBI.clip(pune), {}, 'MaxNDBI');

var stats = {
    MaxNDVI: MaxNDVI.reduceRegion({
        reducer: ee.Reducer.minMax(),
        geometry: pune,
        scale: 30,
        bestEffort: true
    }),
    MaxMNDWI: MaxMNDWI.reduceRegion({
        reducer: ee.Reducer.minMax(),
        geometry: pune,
        scale: 30,
        bestEffort: true
    }),
    MinNDVI: MinNDVI.reduceRegion({
        reducer: ee.Reducer.minMax(),
        geometry: pune,
        scale: 30,
        bestEffort: true
    }),
    MinMNDWI: MinMNDWI.reduceRegion({
        reducer: ee.Reducer.minMax(),
        geometry: pune,
        scale: 30,
        bestEffort: true
    }),
    MinNDBI: MinNDBI.reduceRegion({
        reducer: ee.Reducer.minMax(),
        geometry: pune,
        scale: 30,
        bestEffort: true
    }),
    MaxNDBI: MaxNDBI.reduceRegion({
        reducer: ee.Reducer.minMax(),
        geometry: pune,
        scale: 30,
        bestEffort: true
    })
};

print('MaxNDVI Stats:', stats.MaxNDVI);
print('MaxMNDWI Stats:', stats.MaxMNDWI);
print('MinNDVI Stats:', stats.MinNDVI);
print('MinMNDWI Stats:', stats.MinMNDWI);
print('MinNDBI Stats:', stats.MinNDBI);
print('MaxNDBI Stats:', stats.MaxNDBI);



// Add the minimum values as bands to the image
var SAimageWithMinIndices = datasetNoCloud_Median
    .addBands(MinNDVI)
    .addBands(MinMNDWI)
    .addBands(MaxNDVI)
    .addBands(MaxMNDWI)
    .addBands(MinNDBI)
    .addBands(MaxNDBI);

print("Image with minimum indices", SAimageWithMinIndices);

// Nighttime Light data
var nighttime = ee.ImageCollection('NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG')
    .filter(ee.Filter.date('2024-01-01', '2024-10-01'))
    .select('avg_rad')
    .mean().clip(pune); // Reduce to a single image

Map.addLayer(nighttime, {min: 0.0, max: 60.0}, 'Nighttime');
// Add elevation and slope
var SRTM_30m = ee.Image('USGS/SRTMGL1_003');
var elevation = SRTM_30m.select('elevation');
var slope = ee.Terrain.slope(elevation);
var SAimage = ee.Image.cat([SAimageWithMinIndices, slope, elevation,nighttime]);
 //////Export//////////////////
 Export.image.toDrive({
    image: nighttime,
    description: 'nighttime',
    scale: 30,
    region: pune,
    folder: 'GEE_exports',
    maxPixels: 1e13
  });
  ////Export/////////////
   Export.image.toDrive({
    image: MinNDBI,
    description: 'MinNDBI',
    scale: 30,
    region: pune,
    folder: 'GEE_exports',
    maxPixels: 1e13
  });
  ///////////////////////////////////////
   Export.image.toDrive({
    image: MaxNDBI,
    description: 'MaxNDBI',
    scale: 30,
    region: pune,
    folder: 'GEE_exports',
    maxPixels: 1e13
  });
  //////////////////////////
  Export.image.toDrive({
    image: MinNDVI,
    description: 'MinNDVI',
    scale: 30,
    region: pune,
    folder: 'GEE_exports',
    maxPixels: 1e13
  });
  //////////////////////////
  Export.image.toDrive({
    image: MinMNDWI,
    description: 'MinMNDWI',
    scale: 30,
    region: pune,
    folder: 'GEE_exports',
    maxPixels: 1e13
  });
  ////////////////////
    Export.image.toDrive({
    image: slope,
    description: 'slope',
    scale: 30,
    region: pune,
    folder: 'GEE_exports',
    maxPixels: 1e13
  });
  ////////////////
    Export.image.toDrive({
    image: filter_Sentinel_2,
    description: 'fcc',
    scale: 30,
    region: pune,
    folder: 'GEE_exports',
    maxPixels: 1e13
  });

// Define training bands
var TrainBands = ['B2', 'B3', 'B4', 'B5', 'B8', 'B11', 'MinNDVI', 'MinMNDWI', 'NDVI', 'MNDWI', 'MaxNDVI', 'MaxMNDWI', 'MinNDBI', 'MaxNDBI', 'slope', 'avg_rad'];

// Merge training points
var training_points = waterbodies
    .merge(Forest)
    .merge(Builtup)
    .merge(BarrenLand)
    .merge(ScrubLand)
    .merge(FallowLand)
    .merge(Cropland);

// Create training data
var training_data = SAimage.select(TrainBands).sampleRegions({
  collection: training_points,
  properties: ['CLASS'],
  scale: 30
});

// Merge validation points
var validation_points = V_waterbodies
    .merge(V_Forest)
    .merge(V_Builtup)
    .merge(V_BarrenLand)
    .merge(ScrubLand)
    .merge(V_FallowLand)
    .merge(V_Cropland);

// Define ML models
var models = [
  { name: "Decision Tree", classifier: ee.Classifier.smileCart() },
  { name: "LibSVM", classifier: ee.Classifier.libsvm() },
  { name: "Minimum Distance", classifier: ee.Classifier.minimumDistance() },
  { name: "SMILE CART", classifier: ee.Classifier.smileCart() },
  { name: "SMILE Random Forest", classifier: ee.Classifier.smileRandomForest(100) },
  { name: "SMILE Gradient Tree Boost", classifier: ee.Classifier.smileGradientTreeBoost(50) },
  { name: "SMILE KNN", classifier: ee.Classifier.smileKNN(3) }
  
];

// Classify, validate, visualize, and export for each model
models.forEach(function(model) {
  // Train the classifier
  var classifier = model.classifier.train({
    features: training_data,
    classProperty: 'CLASS',
    inputProperties: TrainBands
  });

  // Classify the image
  var LULC = SAimage.classify(classifier);

  // Add LULC map to the interface
  Map.addLayer(
    LULC.clip(pune),
    { min: 0, max: 7, palette: ['73DFFF', '22A800', 'CA0000', 'D2B48C', '808000', 'E6E600', 'FFFFBE'] },
    model.name + ' LULC'
  );

  // Export the classified image
  Export.image.toDrive({
    image: LULC,
    description: model.name + '_LULC_Classification',
    folder: 'LULC_Classification',
    scale: 30,
    region: pune,
    maxPixels: 1e13,
    crs: 'EPSG:4326'
  });

  // Validate the classification
  var validation = LULC.sampleRegions({
    collection: validation_points,
    properties: ['CLASS'],
    scale: 20
  });

  // Compute error matrix and accuracies
  var testAccuracy = validation.errorMatrix('CLASS', 'classification');
  print(model.name + ' Validation Error Matrix:', testAccuracy);
  print(model.name + ' Overall Accuracy:', testAccuracy.accuracy());
  print(model.name + ' Kappa Coefficient:', testAccuracy.kappa());
  print(model.name + ' Producer\'s Accuracy:', testAccuracy.producersAccuracy());
  print(model.name + ' User\'s Accuracy:', testAccuracy.consumersAccuracy());
});

Export.table.toDrive({
  collection: training_points,
  description: 'Pune_Training_Points',
  fileFormat: 'GeoJSON'
});
Export.table.toDrive({
  collection: validation_points,
  description: 'Pune_validation_points',
  fileFormat: 'GeoJSON'
});

  // Export confusion matrix
 // var confusionMatrixCSV = ee.FeatureCollection([ee.Feature(null, {matrix: testAccuracy.array()})]);
  // Export.table.toDrive({
  //   collection: confusionMatrixCSV,
  //   description: model.name + '_ConfusionMatrix',
  //   fileFormat: 'CSV',
  //   folder: 'GEE_exports'
  // });

  // Export.table.toDrive({
  //   collection: training_points,
  //   description: 'Pune_T_Points',
  //   fileFormat: 'GeoJSON'
  // });

  // Export.table.toDrive({
  //   collection: validation_points,
  //   description: 'pune_Validation_Points',
  //   fileFormat: 'GeoJSON'
  // });

  // Return accuracy metrics as a feature
//   return ee.Feature(null, {
//     Model: model.name,
//     UserAccuracy: testAccuracy.consumersAccuracy(),
//     ProducerAccuracy: testAccuracy.producersAccuracy()
//   });
// });

// // Convert metrics to a feature collection
// var accuracyFeatureCollection = ee.FeatureCollection(accuracyMetrics);

// // Print accuracy metrics
// print('Accuracy Metrics for User and Producer:', accuracyFeatureCollection);

// // Export LULC image
//   Export.image.toDrive({
//     image: S,
//     description: 'slope',
//     scale: 30,
//     region: pune,
//     folder: 'GEE_exports',
//     maxPixels: 1e13
//   });
//Export accuracy metrics as a CSV
// Export.table.toDrive({
//   collection: accuracyFeatureCollection,
//   description: 'User_Producer_Accuracy_Metrics',
//   fileFormat: 'CSV',
//   folder: 'GEE_exports'
// });

// Define the palette for LULC visualization
var palette = ['73DFFF', '22A800', 'CA0000', 'D2B48C', '808000', 'E6E600', 'FFFFBE'];

// Step 6: Train, classify, and visualize LULC for each model
var accuracyMetricsAndLULC = models.map(function(model) {
  // Train the classifier
  var classifier = model.classifier.train({
    features: training_data,
    classProperty: 'CLASS',
    inputProperties: TrainBands
  });

  // Classify the image
  var LULC = filter_Sentinel_2.classify(classifier);
  // Add LULC map to the interface
  Map.addLayer(LULC, {min: 0, max: 10, palette: palette}, model.name + ' LULC');

  // Return LULC visualization feature
  return ee.Feature(null, {
    Model: model.name,
    LULC: LULC
  });
});

// Convert metrics and LULC results to a FeatureCollection
var lulcMetricsFeatureCollection = ee.FeatureCollection(accuracyMetricsAndLULC);

// Export aggregated LULC visualization as a CSV (if needed, though you may want to skip exporting LULC directly)
// Export.table.toDrive({
//   collection: lulcMetricsFeatureCollection,
//   description: 'Model_LULC_Metrics',
//   fileFormat: 'CSV',
//   folder: 'GEE_exports'
// });

print('LULC Metrics for Each Model:', lulcMetricsFeatureCollection);
