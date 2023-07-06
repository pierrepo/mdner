# ANNOTATIONS

## Data sets

The MDverse project aims to make MD data accessible and searchable. By browsing the generalist data repositories Zenodo, Figshare and Open Science Framework (OSF), we have indexed around 250,000 files in almost 2,000 datasets. The table below illustrates the distribution of the data.

<figure class="table" align="center">
<table align="center">
<thead align="center">
  <tr>
    <th align="center">Dataset origin<br></th>
    <th align="center">Number of text data sets<br></th>
  </tr>
</thead>
<tbody>
  <tr>
    <td align="center">Figshare</td>
    <td align="center">913</td>
  </tr>
  <tr>
    <td align="center">Open Science Framework (OSF)</td>
    <td align="center">63</td>
  </tr>
  <tr>
    <td align="center">Zenodo</td>
    <td align="center">1,987</td>
  </tr>
</tbody>
</table>
</figure>

## Structure of an annotation

Each JSON file is structured as a dictionary that includes a list of entities and a list of textual content. The list of textual content comprises a nested list, where the inner list represents the text to be annotated, and a dictionary that maps each token's index to its corresponding entity.

```
{
   "classes":[
      "TEMP",
      "SOFT",
      "STIME",
      "MOL",
      "FFM"
   ],
   "annotations":[
      [
         "POPC bilayer with 30 waters per lipid (17+17), at 300K, simulation using gromacs.",
         {
            "entities":[
               [
                  0,
                  4,
                  "MOL"
               ],
               [
                  21,
                  27,
                  "MOL"
               ],
               [
                  32,
                  37,
                  "MOL"
               ],
               [
                  50,
                  54,
                  "TEMP"
               ],
               [
                  73,
                  80,
                  "SOFT"
               ]
            ]
         }
      ]
   ]
}
```