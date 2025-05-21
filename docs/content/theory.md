# Theory

Here will be the decision tree with some explanation.
For a complete discussion of the theory is provided in the accompanying paper which will be linked here soon. 

## Overview of the sampling scheme

The disaggregation happens by sampling in two "layers" which are then multiplied:

1) the first layer is sampling the aggregate quantity from a the maximum entropy distribution for the provided information. The Following decision tree provides an overview of the principles used in the ```maxent_disaggregation``` package to sample the aggregate quantity.


```{mermaid}
flowchart-elk TD
    %% Define node classes
    classDef decision fill:#e28743,color:black,stroke:none;
    classDef distribution fill:#abdbe3,color:black,stroke:none;
    classDef notimplementednode fill:#eeeee4,color:black,stroke:none;

    MeanDecision{{"Best guess/
    mean available?"}}:::decision
    SDDecision{{"Standard deviation available?"}}:::decision
    BoundsDecision1{{"Bounds available?"}}:::decision
    Uniform("Uniform distribution on [a,b]"):::distribution
    GoBackToStart["☠️ !Game Over!
    We suggest to rethink your problem... 🤓"]:::notimplementednode
    BoundsDecision2{{"Bounds available?"}}:::decision
    Normal("Normal distribution"):::distribution
    UnbiasedMean{{"Prefer unbiased mean?"}}:::decision
    TruncNorm("Truncated Normal 
    (Maximum Entropy distribution)"):::distribution
    LogNorm("LogNormal distribution"):::distribution
    LowerBound0{{"Lower bound = 0?"}}:::decision
    Exponential("Exponential"):::distribution
    NotImplemented["Not Implemented"]:::notimplementednode


    %% Define connections
    MeanDecision -- "no" --> BoundsDecision1
    MeanDecision -- "yes" --> SDDecision
    SDDecision -- "yes" --> BoundsDecision2
    BoundsDecision2 -- "yes" --> UnbiasedMean
    UnbiasedMean -- "yes" --> LogNorm
    UnbiasedMean -- "no" --> TruncNorm
    BoundsDecision2 -- "no" --> Normal
    SDDecision -- "no" --> LowerBound0
    LowerBound0 -- "yes" --> Exponential
    LowerBound0 -- "no" --> NotImplemented
    BoundsDecision1 -- "yes" --> Uniform
    BoundsDecision1 -- "no" --> GoBackToStart
```


2) The second layer is then sampling the disaggregate quantities based, or shares of the aggregate quantity, based on the available information. The following decision tree provides an overview of the principles used in the ```maxent_disaggregation``` package to sample the disaggregate quantities.


```{mermaid}
flowchart-elk TD
    %% Define node classes
    classDef decision fill:#e28743,color:black,stroke:none;
    classDef distribution fill:#abdbe3,color:black,stroke:none;
    classDef explanationnode fill:#eeeee4,color:black,stroke:none;

    MeanDecision{{"Best guess/mean available?"}}:::decision
    SDDecision{{"Standard deviation available?"}}:::decision
    MaxEntDir("Maximum Entropy Dirichlet"):::distribution
    GenDir("Generalised Dirichlet"):::distribution
    NestedDir("Nested Dirichlet"):::distribution
    UniformDir("Uniform Dirichlet"):::distribution
    NestedDir2["Use all available information by dividing the shares into chunks
    and sample them according to the same decision tree"]:::explanationnode

    %% Define connections
    MeanDecision -- "no" --> UniformDir
    MeanDecision -- "yes" --> SDDecision
    MeanDecision -- "paritially" --> NestedDir
    SDDecision -- "no" --> MaxEntDir
    SDDecision -- "yes" --> GenDir
    SDDecision -- "partially" --> NestedDir
    NestedDir --> NestedDir2
    NestedDir2 --> MeanDecision
```
