--- MindsDB ships with a filesystem database called 'files'
--- Each file you uploaded is saved as a table there.
---
--- You can always check the list tables in files as follows:

SHOW TABLES FROM files;

--- These files can be queried as tables, 
--- You just uploaded airline

SELECT t.text,t.sentiment 
FROM files.airline t 
LIMIT 100;

CREATE PREDICTOR mindsdb.reviewdata
FROM files
    (SELECT * FROM files.airline LIMIT 100)
PREDICT sentiment
USING
    model.args={"submodels":[
					{
                        "module": "fdf.FetchDB",
                        "args": {
                            "target_encoder": "$encoders[self.target]",
                            "stop_after": "$problem_definition.seconds_per_mixer"
                        }                        
					}
                ]};

SELECT *
FROM mindsdb.predictors

DESCRIBE PREDICTOR mindsdb.reviewdata.ensemble;

SELECT sentiment
FROM mindsdb.reviewdata
WHERE text='Where are you bloodyful'

SELECT t.text,t.sentiment,m.sentiment
FROM files.airline t
JOIN mindsdb.reviewdata AS m
LIMIT 5;
