
    create or replace view "dev"."main"."growth_rate_overEqual40__dbt_int" as (
      select * from read_csv('./output/growth_rate_overEqual40.csv', auto_detect=True)
      -- if relation is empty, filter by all columns having null values
      
    );
    