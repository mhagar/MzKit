# Chromatogram Types

`.getChromatogramType` returns an integer. The integers correspond to:
0: MASS_CHROMATOGRAM 	
1: TOTAL_ION_CURRENT_CHROMATOGRAM 	
2: SELECTED_ION_CURRENT_CHROMATOGRAM 	
3: BASEPEAK_CHROMATOGRAM 	
4: SELECTED_ION_MONITORING_CHROMATOGRAM 	
5: SELECTED_REACTION_MONITORING_CHROMATOGRAM 	
6: ELECTROMAGNETIC_RADIATION_CHROMATOGRAM 	
7: ABSORPTION_CHROMATOGRAM 	
8: EMISSION_CHROMATOGRAM 	
9: SIZE_OF_CHROMATOGRAM_TYPE 


# Chromatogram Extraction

`aggregateFromMatrix()`: returns a list of lists of floats
`extractXICsFromMatrix()`: returns a list of `MSChromatogram` objects


## Matrix Double
This data structure is extremely poorly documented.
Critically, `MatrixDouble` is used by `extractXICsFromMatrix()` 
and `aggregateFromMatrix()`

I looked through `OpenMS/src/pyOpenMS/tests/unittests/test000.py` to 
figure out what's going on.

To summarize, this object can be created using `fromNdArray`, which takes
a **numpy array of shape (n_ranges, 4)** (`[mz_min, mz_max, rt_min, rt_max]`))

That is, a row for each range to be extracted.
For example, to extract a TIC for glucose (181 m/z)
```
    ranges_matrix = pyopenms.MatrixDouble.fromNdArray(
        np.array(
            [
                [180.5, 181.5, 0.0, 400.0]
            ]
        )
    )
    
    exp.extractXICsFromMatrix(
        ranges_matrix,
        ms_level=2,
        mz_agg=b'sum'  # This means SUM the m/z signals
    )
```

## CRITICAL
Options for `mz_agg` are:
```
b"sum", b"max", b"min", b"mean"
```

# Mass Trace Detection

Again, poor documentation. See the following:
```
MTD = oms.MassTraceDetection()
mass_traces = [oms.Kernel_MassTrace()]
MTD.run(
    exp,             # MS Experiment
    mass_traces,     # Container to save mass traces
    10000,           # Maximum number of traces
)
```


