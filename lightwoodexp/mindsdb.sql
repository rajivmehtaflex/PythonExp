--- MindsDB ships with a filesystem database called 'files'
--- Each file you uploaded is saved as a table there.
---
--- You can always check the list tables in files as follows:

SHOW TABLES FROM files;

--- These files can be queried as tables, 
--- You just uploaded review

;

SELECT * FROM files.review LIMIT 100

drop table files.review

DROP PREDICTOR mindsdb.reviewdata

CREATE PREDICTOR mindsdb.reviewdata
FROM files
    (SELECT * FROM files.review LIMIT 100)
PREDICT positive
USING
    model.args={"submodels":[
					{
						'module': 'faker.Faker',
						'args': {
							'stop_after': '$problem_definition.seconds_per_mixer',
							'dtype_dict': '$dtype_dict',
							'target': '$target',
							'target_encoder': '$encoders[self.target]'
						}
					}
                ]};


SELECT positive
FROM mindsdb.reviewinfo
WHERE revewtext='ok let us do it!!!'


DESCRIBE PREDICTOR mindsdb.reviewdata.features;


SELECT *
FROM mindsdb.predictors
WHERE name='home_rentals_model';