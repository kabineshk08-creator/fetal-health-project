# Upload & Data Management Features

## Overview

The Fetal Health Classification system now includes comprehensive upload and preprocessing capabilities:

### 1. **CSV Data Upload with Automatic Preprocessing**
- Upload CTG data files for automatic preprocessing
- Individual record output generation (one output per record)
- Missing value handling
- Data scaling and normalization
- Preprocessing status messages

### 2. **Model File Upload**
- Upload trained model files (.h5 for Keras/TensorFlow, .pkl for scikit-learn)
- Automatic model reloading into the system
- Admin-only access
- Real-time model status updates

### 3. **Graph-Based Image Upload**
- Upload CTG signal graphs and charts (PNG, JPG, JPEG, GIF)
- Automatic graph validation using edge detection
- Ensures only valid graph-based data is accepted
- Quality validation before storage

## Endpoints & Routes

### Web Interface

#### `/upload` - Upload Page
- **Method:** GET, POST
- **Authentication:** Required (login_required)
- **Description:** Main upload interface for all file types
- **Features:**
  - File type selection dropdown
  - File input with validation
  - Real-time results display
  - Sample record preview

### API Endpoints

#### `/api/upload_data` - Data Preprocessing API
- **Method:** POST
- **Authentication:** Required
- **Content-Type:** multipart/form-data
- **Parameters:**
  - `file`: CSV file to preprocess
- **Response:**
```json
{
  "success": true,
  "message": "✅ Data preprocessed successfully",
  "filename": "processed_data_20260505_120000.csv",
  "rows": 126,
  "columns": 21,
  "record_outputs": [
    {
      "record_id": 1,
      "features": {"baseline": 120.5, ...},
      "status": "preprocessed"
    }
  ],
  "total_records": 126
}
```

#### `/api/upload_image` - Image Upload & Validation API
- **Method:** POST
- **Authentication:** Required
- **Content-Type:** multipart/form-data
- **Parameters:**
  - `file`: Image file (PNG, JPG, JPEG, GIF)
- **Response:**
```json
{
  "success": true,
  "message": "✅ Image successfully validated and uploaded",
  "filename": "ctg_graph_20260505_120000.png",
  "validation": "✅ Valid graph image (edge density: 15.42%)",
  "path": "/path/to/uploads/ctg_graph_20260505_120000.png"
}
```

#### `/api/upload_model` - Model Upload API
- **Method:** POST
- **Authentication:** Required (Admin only)
- **Content-Type:** multipart/form-data
- **Parameters:**
  - `file`: Model file (.h5, .pkl)
- **Response:**
```json
{
  "success": true,
  "message": "✅ Model uploaded successfully",
  "filename": "ann_model.h5",
  "models_loaded": ["ann", "cnn", "scaler"]
}
```

## Data Preprocessing Details

### CSV Input Format
- **Required:** 21 CTG features
- **Optional:** Fetal health target column (automatically removed if present)
- **Format:** Standard CSV with headers

### Preprocessing Steps
1. **Missing Value Handling**
   - Strategy: Mean imputation
   - Output: Dataset with all missing values filled

2. **Feature Scaling**
   - Method: StandardScaler (default)
   - Preserves feature distributions
   - Produces zero-mean, unit-variance features

3. **Individual Record Output**
   - Each row processed individually
   - Status tracking for each record
   - Feature extraction and normalization per record

### Output Format
- **File:** `processed_data_YYYYMMDD_HHMMSS.csv`
- **Location:** `/uploads/` directory
- **Content:** Preprocessed data with normalized features
- **Records:** Individual output for each input row

## Graph-Based Image Validation

### Validation Criteria
The system validates uploaded images to ensure they contain actual graph-based data:

1. **Edge Density Check**
   - Too sparse (< 5%): Rejected - not enough structure
   - Valid range: 5% - 80%
   - Too dense (> 80%): Rejected - too noisy

2. **Structure Detection**
   - Analyzes edge patterns to identify:
     - Lines and axes
     - Gridlines
     - Signal patterns
     - Chart structures

3. **Quality Metrics**
   - Contrast levels
   - Pattern consistency
   - Structure coherence

### Accepted Formats
- PNG (Portable Network Graphics)
- JPG/JPEG (Joint Photographic Experts Group)
- GIF (Graphics Interchange Format)

### Maximum File Size
- 16MB per file

## Features & Capabilities

### ✅ Automatic Preprocessing
- Data validation before processing
- Error handling and recovery
- Batch processing support
- Progress logging

### ✅ Individual Record Output
- One output per input record
- Feature-level details
- Processing status per record
- Supports batch operations

### ✅ Graph Validation
- AI-based image analysis
- Automatic quality assessment
- Rejection of invalid data
- Edge detection algorithm

### ✅ Audit Trail
- User tracking (who uploaded what)
- Timestamp logging
- File naming with timestamps
- Operation logging

## Usage Examples

### Upload CSV Data (Web Interface)
1. Navigate to Dashboard → Upload Data
2. Select "📊 CSV Data (for preprocessing)"
3. Choose your CSV file
4. Click "Upload File"
5. View results and individual record outputs

### Upload via API (Python)
```python
import requests

# Upload CSV data
files = {'file': open('data.csv', 'rb')}
response = requests.post(
    'http://localhost:5000/api/upload_data',
    files=files,
    headers={'Authorization': 'Bearer YOUR_TOKEN'}
)
print(response.json())
```

### Upload Graph-Based Image (Web Interface)
1. Navigate to Dashboard → Upload Data
2. Select "🖼️ CTG Image (graph-based data)"
3. Choose your graph image
4. System validates automatically
5. Success: Image stored; Failure: Get validation feedback

### Upload Model (Admin)
1. Navigate to Dashboard → Upload Data
2. Select "🤖 Model File (.h5, .pkl)"
3. Choose trained model
4. System reloads models into memory
5. Confirmation message shows loaded models

## System Messages

### Success Messages
- `✅ Data preprocessed successfully` - CSV preprocessing completed
- `✅ Image successfully validated and uploaded` - Image passed validation
- `✅ Model uploaded successfully` - Model file saved and loaded

### Error Messages
- `❌ No file selected` - No file provided
- `❌ Invalid file type` - Wrong file extension
- `❌ CSV file is empty` - Empty CSV file
- `❌ Image does not contain graph-like structures` - Not a graph image
- `❌ Image appears to be too noisy` - Image quality too low

## Database Integration

### Prediction Records
Each upload is optionally logged with:
- User ID
- Upload timestamp
- File name
- Upload type (data/model/image)
- Processing status
- Results summary

## Security Considerations

1. **File Validation**
   - Extension checking
   - File size limits (16MB)
   - Content type validation

2. **Access Control**
   - User authentication required
   - Admin-only for model uploads
   - User data isolation

3. **Data Handling**
   - Secure filename generation
   - Temporary file cleanup
   - No sensitive data logging

## Performance

- **CSV Processing:** O(n) where n = number of records
- **Image Validation:** ~1-2 seconds per image
- **Model Loading:** Depends on model size (~1-5 seconds)
- **Concurrent Uploads:** Limited by server resources

## Troubleshooting

### CSV Upload Issues
| Problem | Solution |
|---------|----------|
| "Invalid CSV file" | Check CSV format and headers |
| "CSV file is empty" | Ensure CSV has data rows |
| "21 features expected" | CSV must have exactly 21 features |

### Image Upload Issues
| Problem | Solution |
|---------|----------|
| "Too sparse" | Use higher contrast graphs |
| "Too noisy" | Reduce image noise/artifacts |
| "Invalid format" | Use PNG, JPG, JPEG, or GIF |

### Model Upload Issues
| Problem | Solution |
|---------|----------|
| "Invalid model file" | Ensure .h5 or .pkl format |
| "Access denied" | Admin privileges required |
| "Model not loading" | Check model file integrity |

## Future Enhancements

- [ ] Batch upload for multiple files
- [ ] Upload progress tracking
- [ ] Advanced image preprocessing
- [ ] Real-time dashboard updates
- [ ] Custom preprocessing pipelines
- [ ] Automated model training on upload
- [ ] Download processed data
- [ ] Upload history dashboard

## Support

For issues or questions:
1. Check logs at `/logs/upload.log`
2. Review system messages in UI
3. Contact system administrator
4. Report bugs with file samples (if possible)
