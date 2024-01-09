# **EarthQuake Damage Prediction**
------------------------------
## **Business Case-Study**

Earthquakes are natural disasters with the potential to cause widespread devastation, leading to loss of life, property damage, and disruption of
essential services. Traditional seismic design and retrofitting methods have limitations in predicting and preventing the full extent of damage.
Hence, there is a critical need for innovative solutions that leverage advanced technologies to enhance predictive capabilities of earthquake damage.
In recent years, the frequency and intensity of earthquakes have posed a significant threat to urban infrastructure, causing substantial
economic losses and human suffering.

So, This business case study explores the development and implementation of an advanced earthquake damage prediction system aimed at mitigating the
impact of seismic events on buildings infrastructure in earthquake-prone regions. The primary objective of this business case is to propose the
development and deployment of an earthquake damage prediction system that utilizes six different types of machine learning algorithms including:
Logistic Regression, K-Nearest Neighbors, Decision Trees, Random Forest,
Gradient Boosting, XG-Boosting.

Benefit from this model: Improved Urban Planning, Enhanced Public Safety, Reduced Economic Losses caused by earthquakes.

--------
## **Dataset Description**

This dataset consists total 40 features with 260601 rows.

**Target explained** here:

1) Damage_grade: This presents the level of damage grade affected at the time of earthquake.(1,2,3 represents low damage, medium amount of damage,almost complete destruction respectively)

**Others Features** are explained here:

1) building_id: It represents each building units.

2) geo_level_1_id: This is the highest level of geographic region classification. It can have values in the range of 0 to 30. Each value typically represents a broad region or country.

3) geo_level_2_id: This is a more detailed sub-region within the geo_level_1_id. It can have values in the range of 0 to 1427, which provides more specific information about the region where the building is located.

4) geo_level_3_id: This is the most specific sub-region classification. It can have values in the range of 0 to 12567, offering the most detailed information about the geographic location of the building.

5) count_floors_pre_eq: Number of floors in the building before the earthquake.

6) age: Age of the building in years.

7) area_percentage: Normalized area of the building footprint.

8) height_percentage: Normalized height of the building footprint.

9) land_surface_condition: Surface condition of the land where the building was built. Possible values:(n: Normal Surface Condition, o: Poor Surface Condition, t: Unusual Surface Condition)

10) foundation_type: Type of foundation used in the building. Possible values:(h: Hard or Concrete, i: Similartohard, r: Rubble stone, u: Bamboo or Timber, w: Other or Wooden)

11) roof_type: Type of roof used in building. Possible values:(n: No roof, q: Quasi-flat roof., x: Traditional roof structure)

12) ground_floor_type: Type of the ground floor. Possible values:(f:Floorslab, m: Mud, v: Other, x: Timber, z: Bamboo)

13) other_floor_type: Type of constructions used in higher than the ground floors (except roof). Possible values:(j: Jackets, q: Quasi-adobe, s: Wooden or Timber, x: Unknown)

14) position: Position of the building while construction. Possible values:(j: Jutting, o: Onesideattached, s: Semi-detached or Attached-2 side, t: Terraced or Linear)

15) plan_configuration: Building plan configuration. Possible values:(a:Rectangular, c: L-shaped, d: T-shaped, f: Wing-shaped, m: Multiple or Complex, n: Square-shaped, o: Other or Unique,q: H-shaped, s: U-shaped, u: E-shaped)

16) has_superstructure_adobe_mud: Flag variable that indicates if the superstructure was made of Adobe/Mud.

17) has_superstructure_mud_mortar_stone: Flag variable that indicates if the superstructure was made of Mud Mortar-Stone.

18) has_superstructure_stone_flag: Flag variable that indicates if the superstructure was made of Stone.

19) has_superstructure_cement_mortar_stone: Flag variable that indicates if the superstructure was made of Cement Mortar-Stone.

20) has_superstructure_mud_mortar_brick: Flag variable that indicates if the superstructure was made of Mud Mortar-Brick.

21) has_superstructure_cement_mortar_brick: Flag variable that indicates if the superstructure was made of Cement Mortar - Brick.

22) has_superstructure_timber: Flag variable that indicates if the superstructure was made of Timber.

23) has_superstructure_bamboo: Flag variable that indicates if the superstructure was made of Bamboo.

24) has_superstructure_rc_non_engineered: Flag variable that indicates if the superstructure was made of non-engineered reinforced concrete.

25) has_superstructure_rc_engineered: Flag variable that indicates if the superstructure was made of engineered reinforced concrete.

26) has_superstructure_other: Flag variable that indicates if the superstructure was made of any other material.

27) legal_ownership_status: Legal ownership status of the land where building was built. Possible values:(a: Own or Private ownership, r: Rented, v: Government-owned, w: Institutional ownership)

28) count_families: Number of families that live in the building.

29) has_secondary_use: Flag variable that indicates if the building was used for any secondary purpose.

30) has_secondary_use_agriculture: Flag variable that indicates if the building was used for agricultural purposes.

31) has_secondary_use_hotel: Flag variable that indicates if the building was used as a hotel.

32) has_secondary_use_rental: Flag variable that indicates if the building was used for rental purposes.

33) has_secondary_use_institution: Flag variable that indicates if the building was used as a location of any institution.

34) has_secondary_use_school: Flag variable that indicates if the building was used as a school.

35) has_secondary_use_industry: Flag variable that indicates if the building was used for industrial purposes.

36) has_secondary_use_health_post: Flag variable that indicates if the building was used as a health post.

37) has_secondary_use_gov_office: Flag variable that indicates if the building was used has a government office.

38) has_secondary_use_use_police: Flag variable that indicates if the building was used as a police station.

39) has_secondary_use_other: Flag variable that indicates if the building was secondarily used for other purposes.

-----------
### **Analysis of categorical features**

![image](https://github.com/anjanikmr39/EarthQuake-Damage-Prediction/assets/67219753/91e56ba1-8e94-4777-8b11-94d3c499bdd2)

We are considering damage grade into 3 categories (0 to 1) count as 1, (1 to 2) count as 2, and (2 to 3) count as 3 from barplot.

We are Dropping 22 features from categorical data because:

1) 'count_floors_pre_eq': We analyse that  1st floor building have more damage than 6/7 th floor building which does not make any sense from barplot.

2) 'land_surface_condition': We analyse that All Unusual,Poor and Normal have almost same damgae grade level from barplot.

3) 'position': We analyse that All Twosideattached, Linear/rows, Single and Onesideattached have almost same damgae grade level from barplot.

4) 'has_superstructure_adobe_mud': We analyse that 0 and 1 have almost same damgae grade level from barplot.

5) 'has_superstructure_stone_flag': We analyse that 0 and 1 have almost same damgae grade level from barplot.

6) 'has_superstructure_mud_mortar_brick': We analyse that 0 and 1 have almost same damgae grade level from barplot.

7) 'has_superstructure_timber' : We analyse that 0 and 1 have almost same damgae grade level from barplot.

8) 'has_superstructure_bamboo': We analyse that 0 and 1 have almost same damgae grade level from barplot.

9) 'has_superstructure_other': We analyse that 0 and 1 have almost same damgae grade level from barplot.

10) 'has_secondary_use': We analyse that 0 and 1 have almost same damgae grade level from barplot.

11) 'has_secondary_use_agriculture': We analyse that 0 and 1 have almost same damgae grade level from barplot.

12) 'has_secondary_use_industry': We analyse that 0 and 1 have almost same damgae grade level from barplot.

13) 'has_secondary_use_use_police': We analyse that 0 and 1 have almost same damgae grade level from barplot.

14) 'has_secondary_use_other': We analyse that 0 and 1 have almost same damgae grade level from barplot.

15) 'has_superstructure_cement_mortar_stone': From count plot, We analyse that 0(98.18% ) and 1(1.82%), so we are considering it as constant feature.

16) 'has_secondary_use_rental': From count plot, Here also we analyse data biasness towards  0. It have 0(99.19%) and 1(0.81%). So this one is also constant feature.

17) 'has_secondary_use_institution': From count plot, It have 0(99.91%) and 1(0.09%) so this is also treated as constant feature.

18) 'has_secondary_use_school': From count plot, It have 0(99.96%) and 1(0.04%) so this is also treated as constant feature.

19) 'has_secondary_use_health_post': From count plot, It have 0(99.98%) and 1(0.02%) so this is also treated as constant feature.

20) 'has_secondary_use_gov_office': From count plot, It have 0(99.99%) and 1(0.01%) so this is also treated as constant feature.

21) & 22) 'has_superstructure_rc_non_engineered' & 'has_superstructure_rc_engineered': We analyse that total number of (245464/260601) rows have same value of 0 in engineered and non-engineered features. So we cannot consider these features since it does not make any sense if engineered is 0 then non engineered must be 1, but here this logic does not apply. Therefore we are removing these two.
----------
### **Analysis of Numerical features**

![image](https://github.com/anjanikmr39/EarthQuake-Damage-Prediction/assets/67219753/fe667197-4d14-4bf9-bd80-f66311429ba3)


![image](https://github.com/anjanikmr39/EarthQuake-Damage-Prediction/assets/67219753/00d49f04-6696-40de-9a71-6e5b3faa69e2)

We did not found any significant information from the barplot.
From histogram we can say that age,area_percentage and height_percentage are showing skewness in nature as we can see their skewness value
and kurtosis value which does not lies between -1 to +1. So for this we have to check for outliers from boxplot and then remove it.

--------
## **Model Comparision Report**

```
+---------------------+--------------------+-----------------+
| Model Name          | Testing Accuracy   | Ranking order   |
+=====================+====================+=================+
| XG-Boosting         | 73%                | 1st(BEST)       |
+---------------------+--------------------+-----------------+
| Random Forest       | 71%                | 2nd             |
+---------------------+--------------------+-----------------+
| Decision Tree       | 70%                | 3rd             |
+---------------------+--------------------+-----------------+
| Gradient Boosting   | 68%                | 4th             |
+---------------------+--------------------+-----------------+
| K-Nearest Neighbour | 68%                | 5th             |
+---------------------+--------------------+-----------------+
| Logistic Regression | 58%                | 6th             |
+---------------------+--------------------+-----------------+

```

--------
## **Suggestions to the Seismologists**
1) Foundation type should be similar to hard:
This might suggest that the foundation is similar to a hard or concrete foundation but not exactly the same. It could include variations or alternative materials that share characteristics with traditional hard foundations.

2) Roof type should be traditional:
This suggests that the building has a traditional roof structure. Traditional roofs can come in various styles, such as gable, hip, mansard, or shed roofs, depending on the architectural design and regional building practices.

3) Ground floor type should be others:
It represents an option not covered by the specific categories listed. It could include various materials or construction methods for the ground floor that do not fit into the other categories such as Floorslab, Mud, Timber, Bamboo.

4) Other floor type should be wooden:
It suggests that the construction of the higher floors involves the use of wood or timber. Wooden construction is common in many building practices, providing a lightweight and versatile material for framing and structural support.

5) Plan configuration should be L shaped:
The building plan is in the shape of an "L". L-shaped configurations are often used to provide variations in space and create different zones within a building.

6) Legal ownership status should be private:
This indicates that the land where the building is constructed is privately owned.

7) Age of building:
After 15 years there would be high chance of more damage so we have to do renovation of the building.

--------
## **Challenges faced**
We did all the steps such as Basic info, EDA, Preprocessing, Model creation. But we have to face challenges mostly in EDA part, because there are total 40 features in which we have to perform feature extraction by analysing the EDA part from different graphs. Atlast we finally reduced 40 features into 17 features. And then apply different six models and shows their Accuracy in Model Comparision Report.

-----------

## **More information**
Visit this python file for more detailed analysis [EquakeDamagePred.ipynb](https://github.com/anjanikmr39/EarthQuake-Damage-Prediction/blob/master/EquakeDamagePred.ipynb).
