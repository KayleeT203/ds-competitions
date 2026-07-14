{{ config(
    materialized='external',
    format='csv',
    location='./output/growth_rate_overEqual40.csv'
) }}

WITH clients AS (
    PIVOT {{ ref('net_value_clients') }}
    ON relative_year
    USING first(net_sales_yen)
),
clients_growth_rate AS (
    SELECT company, ROUND((((CurrentYear/Prior4Year)-1)*100), 2) as growth_rate_2022to2026 
    FROM clients
    WHERE growth_rate_2022to2026 >= 40
)
SELECT * FROM clients_growth_rate