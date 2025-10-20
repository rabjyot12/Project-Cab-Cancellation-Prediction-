# 🚖 Cab Cancellation Prediction using Machine Learning

## 📘 Overview
This project predicts whether a booked cab ride will be **cancelled or completed** based on various factors such as booking time, travel type, booking source, and location details.  
The goal is to help cab companies reduce last-minute cancellations, improve driver utilization, and optimize ride allocation.

---

## 🧠 Problem Statement
Cab cancellations cause operational inefficiencies and loss of revenue for cab aggregators.  
This project builds a **classification model** that predicts the likelihood of a ride being cancelled, enabling companies to take proactive measures (like confirming with customers or reassigning drivers early).

---

## 📊 Dataset
The dataset (`YourCabs.csv`) contains booking-level information such as:
| Feature | Description |
|----------|--------------|
| `user_id` | Unique ID of the customer |
| `vehicle_model_id` | Vehicle category booked |
| `travel_type_id` | Type of travel (City / Outstation / One-way etc.) |
| `booking_created` | Timestamp of booking creation |
| `from_date` | Scheduled start time of the ride |
| `online_booking`, `mobile_site_booking` | Booking source |
| `Car_Cancellation` | Target variable (1 = Cancelled, 0 = Not Cancelled) |

Total Records: ~43,000  
Target Variable Distribution: **7.2% cancelled**, **92.8% completed**

---

## ⚙️ Data Preprocessing
- Dropped irrelevant columns: `from_city_id`, `to_city_id`
- Handled missing values (removed rows with missing datetime)
- Converted date fields to datetime objects
- Feature engineering:
  - `ride_hour`, `ride_dayofweek`
  - `booking_hour`, `booking_dayofweek`
  - `booking_advance_hours`
  - `is_same_day_booking`
  - `booking_nature` (Urgent / SameDay / Regular / Advance)
- Outliers detected using IQR method
- Encoded categorical fields
- Handled class imbalance using `class_weight='balanced'`

---

## 🧮 Model Building
The main model used:
```python
RandomForestClassifier(class_weight='balanced', random_state=42)
