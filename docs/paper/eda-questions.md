Checking the interaction network data

Six things worth checking before we trust this data. The pipeline is described in dataset-construction.md.



1. Do grasses have pollinators?

They shouldn't. Grasses, sedges and rushes are wind-pollinated, and they don't produce nectar and don't recruit insects. 

Check 1: Poaceae, Cyperaceae and Juncaceae should sit at the bottom of the degree distribution. Well below Asteraceae, Lamiaceae, Fabaceae and Rosaceae.





Do so across Tier A and Tier A+B

If grasses turn up with lots of "pollinators", it means records of insects simply sitting on a plant are being counted as flower visits. That's the thing the A/B tier split is supposed to prevent, so check it for Tier A and Tier A+B separately. 

2. Does the fauna look like North America?

Check 2: Break the edges down by pollinator order and by plant family. We'd expect bees and wasps first, then butterflies and moths, then flies, beetles, and a small tail of hummingbirds.

If orders that don't visit flowers show up large (true bugs, spiders, grasshoppers) something is leaking in. 

3. iNaturalist v. other sources?

For each contributing source, how many records, how many edges, and how many edges only that source knows about? How many edges are unique to non iNat?

If most of the network rests on a single source, then what we've built describes that source's sampling habits rather than the region's ecology. This also tells us which sources are big enough to hold out as a test set later.

4. Is recent data different from old data?

Plot records per year, then plot the share of each pollinator order per year.

Citizen science took off around 2015, and it photographs different things than museums collect. More butterflies, less bees. If the composition shifts sharply over time, then splitting the data by year isn't really a test of predicting the future. It's a test of predicting a different kind of record. We need to know that before we use a temporal split.

5. Where do the genus-level records sit?

What fraction of nodes are identified only to genus, broken down by family? And how do genus-level nodes compare to species-level ones in degree?

Genus-only identifications should cluster in groups that are genuinely hard to identify from a photograph (small flies, asters). If they're spread evenly across everything, that's more likely a name-matching failure in the pipeline than an identification limit in the field.

6. Are the interactions real?

Look for relationships we know exist: squash bees (Peponapis, Xenoglossa) on Cucurbita, yucca moths (Tegeticula) on Yucca, specialist Andrena on their host families. Then pull 20 edges at random and see whether they're believable.

The first part confirms real ecological structure survived the pipeline. The second gives us a rough
sense of how noisy the labels are, which matters because the models will be judged against them.



What to hand back: a short write-up, a figure or table for each question, and for each one a plain
verdict — looks right, or looks wrong and here's the step I'd suspect.