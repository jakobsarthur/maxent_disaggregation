# Theory

For a complete discussion of the theory is provided in the accompanying paper which will be linked here soon...

[DOI](...)

## Background: Uncertainty propagation involving data disaggregation

The goal of `maxent_disaggregation` is to provide an easy to use `Python` tool 
that helps you with uncertainty propagation when data disaggregation is involved. Data
disaggregation usually involves splitting one data point into several
disaggregates using proxy data. It is a common problem in many different
research disciplines.


```{mermaid}
flowchart-elk TD
    %% Define node classes
    classDef Aggregate fill:#eeeee4,color:black,stroke:none;
    classDef DisAgg1 fill:#abdbe3,color:black,stroke:none;
    classDef DisAgg2 fill:#e28743,color:black,stroke:none;
    classDef DisAgg3 fill:#abdbe3,color:black,stroke:none;

    agg(("      $Y_0$      ")):::Aggregate
    disagg1(("$Y_1=x_1 Y_0$")):::DisAgg1
    disagg2(("$Y_2=x_2 Y_0$")):::DisAgg2
    disagg3(("$Y_3=x_3 Y_0$")):::DisAgg3
   

    %% Define connections
    agg  --> disagg1
    agg  --> disagg2
    agg  --> disagg3
```

Data disaggregation usually involves an aggregate flow $Y_0$, which is
known, such as the total amount of steel manufactured in a given time
and geography. What we do not know but are interested in are the $K$
disaggregate flows $Y_1,...,Y_K$, such as the different end-use sectors
where the manufactured steel ends up. Even though we do not know the
values of $Y_1, ..., Y_K$, our model structures commonly demands that
the individual $Y_i$’s need to sum to the known aggregate flow $Y_0$ to
respect the mass, energy, stoichiometric or economic balance of the
model

$$
  Y_0 = \sum_{i=1}^{K} Y_i 
$$

This equation, also called an *accounting identity* introduces
dependencies/correlations between the individual disaggregate flows
$Y_i$.

To get estimates for the disaggregate flows, one usually looks for proxy
data. Those proxy data are used to calculate shares (ratios/fractions)
of the respective disaggregate units $x_1, ..., x_K$. To
allocate the entire aggregate flow without leaving any residual (thus to
respect the system balance), those fractions need to sum to one:

$$    \sum_{i=1}^{K} x_i = 1 $$

Disaggregate flows are calculated as

$$ y_i = x_i y_0,  \forall i \in \{1,...,K\}.     $$

## Sampling disaggregates

The `maxent_disaggregation` package generates a random sample of disaggregates
based on the information provided, which need not be complete as is often the
case. The aggregate and the shares are sampled independently and then multiplied
togehter. The distribution from which to sample is determined internally based on
the information provided by the user, following a decision tree that is mostly
based on the principle of Maximum Entropy (MaxEnt):


### Choice of distribution for the aggregate

1) The aggregate quantity is sampled from the maximum entropy distribution for the provided information. The Following decision tree provides an overview of the principles used in the ```maxent_disaggregation``` package to sample the aggregate quantity.


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


2) The disaggregate quantities, or shares of the aggregate quantity, are sampled based on the available information. The following decision tree provides an overview of the principles used in the ```maxent_disaggregation``` package to sample the disaggregate quantities.


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
    hybridDir("Hybrid Dirichlet"):::distribution
    UniformDir("Uniform Dirichlet"):::distribution
    
    %% Define connections
    MeanDecision -- "no" --> UniformDir
    MeanDecision -- "yes" --> SDDecision
    MeanDecision -- "paritially" --> hybridDir
    SDDecision -- "no" --> MaxEntDir
    SDDecision -- "yes" --> GenDir
    SDDecision -- "partially" --> hybridDir
```
