import numpy as np
from matplotlib import pyplot as plt

from landlab import RasterModelGrid, imshow_grid
from landlab.components import (
    ChannelProfiler,
    ChiFinder,
    FlowAccumulator,
    SteepnessFinder,
    StreamPowerEroder,
    LinearDiffuser,
)
from landlab.io import write_esri_ascii

def wrapper(D = 0.5,K_sp = 1.0e-5,m_sp = 0.5,n_sp = 1.0):
    # Grid and model parameters
    number_of_rows = 120
    number_of_columns = 120
    dxy = 100

    # Create a Raster Model Grid
    mg1 = RasterModelGrid((number_of_rows, number_of_columns), dxy)
    
    # Add initial noise to topography
    np.random.seed(56)
    mg1_noise = (np.random.rand(mg1.number_of_nodes) / 1000.0)
    
    z1 = mg1.add_zeros("topographic__elevation", at="node")
    z1 += mg1_noise

    # Set watershed boundary condition
    mg1.set_watershed_boundary_condition("topographic__elevation")

    # Time stepping parameters
    tmax = 1e6
    dt = 1000
    total_time = 0.0

    # Time array for stepping
    t = np.arange(0, tmax, dt)

    # Stream power eroder parameters
    K_sp = 1.0e-5
    m_sp = 0.5
    n_sp = 1.0

    # Components initialization
    frr = FlowAccumulator(mg1, flow_director='FlowDirectorD8')
    spr = StreamPowerEroder(mg1, K_sp=K_sp, m_sp=m_sp, n_sp=n_sp, threshold_sp=0.0)
    
    # D = 0.5  # Diffusivity for linear diffusion
    ld = LinearDiffuser(mg1, linear_diffusivity=D, deposit=False)

    theta = m_sp / n_sp  # Reference concavity for steepness finder
    sf = SteepnessFinder(mg1, reference_concavity=theta, min_drainage_area=1000.0)

    cf = ChiFinder(mg1, min_drainage_area=1000.0, reference_concavity=theta, use_true_dx=True)

    # Uplift rate (uniform uplift)
    uplift_rate = np.ones(mg1.number_of_nodes) * 0.0001

    # Time stepping loop
    for ti in t:
        # Uplift applied only on core nodes
        z1[mg1.core_nodes] += uplift_rate[mg1.core_nodes] * dt
        ld.run_one_step(dt)
        frr.run_one_step()
        spr.run_one_step(dt)
        
        total_time += dt
        print(f"Total time: {total_time} years")

    # Plot the final topography
    imshow_grid(
        mg1,
        "topographic__elevation",
        grid_units=("m", "m"),
        var_name="Elevation (m)"
    )
    title_text = f"$K_{{sp}}$ = {K_sp}; $D$ = {D}; time = {total_time} yr; $dx$ = {dxy} m"
    plt.title(title_text)
    plt.show()

    # Find and print maximum elevation
    max_elev = np.max(z1)
    print("Maximum elevation is:", max_elev)

    # Plot topographic slope
    imshow_grid(
        mg1,
        "topographic__steepest_slope",
        grid_units=("m", "m"),
        var_name="Topographic slope (m/m)"
    )
    plt.title(title_text)
    plt.show()

    # Compute and print mean slope on core nodes
    mean_slope = np.average(mg1.at_node["topographic__steepest_slope"][mg1.core_nodes])
    print("Mean slope is:", mean_slope)

    # Log-log plot of drainage area vs. slope
    plt.loglog(
        mg1.at_node["drainage_area"][mg1.core_nodes],
        mg1.at_node["topographic__steepest_slope"][mg1.core_nodes],
        "b."
    )
    plt.xlabel("Drainage area (m$^2$)")
    plt.ylabel("Topographic slope")
    plt.title(title_text)
    plt.show()

    # Channel profiling
    prf = ChannelProfiler(
        mg1,
        number_of_watersheds=1,
        main_channel_only=True,
        minimum_channel_threshold=dxy**2
    )
    prf.run_one_step()

    # Figure 1: Profile along channels (1-D representation)
    plt.figure(1)
    prf.plot_profiles(
        xlabel='Distance upstream (m)',
        ylabel='Elevation (m)',
        title=title_text
    )
    plt.show()

    # Figure 2: Map view of channel profiles
    plt.figure(2)
    prf.plot_profiles_in_map_view()
    plt.show()

    # Figure 3: Log-log plot of drainage area vs. slope for each channel segment
    plt.figure(3)

    for i, outlet_id in enumerate(prf.data_structure):
        for j, segment_id in enumerate(prf.data_structure[outlet_id]):
            label = f"Channel {i+1}" if j == 0 else '_nolegend_'
            segment = prf.data_structure[outlet_id][segment_id]
            profile_ids = segment["ids"]
            color = segment["color"]
            plt.loglog(
                mg1.at_node["drainage_area"][profile_ids],
                mg1.at_node["topographic__steepest_slope"][profile_ids],
                '.',
                color=color,
                label=label
            )
    plt.legend(loc="lower left")
    plt.xlabel("Drainage area (m$^2$)")
    plt.ylabel("Topographic slope (m/m)")
    plt.title(title_text)
    plt.show()

    # Calculate steepness index
    sf.calculate_steepnesses()

    # Figure 6: Plot steepness index against distance upstream
    plt.figure(6)
    for i, outlet_id in enumerate(prf.data_structure):
        for j, segment_id in enumerate(prf.data_structure[outlet_id]):
            label = f"Channel {i+1}" if j == 0 else '_nolegend_'
            segment = prf.data_structure[outlet_id][segment_id]
            profile_ids = segment["ids"]
            distances = segment["distances"]
            color = segment["color"]
            plt.plot(
                distances,
                mg1.at_node["channel__steepness_index"][profile_ids],
                'x',
                color=color,
                label=label
            )
    plt.xlabel("Distance upstream (m)")
    plt.ylabel("Steepness index")
    plt.legend(loc="upper left")
    plt.title(f"$K_{{sp}}$ = {K_sp}; $D$ = {D}; time = {total_time} yr; $dx$ = {dxy} m; concavity = {theta}")
    plt.show()

    # Figure 7: Map view of steepness index
    plt.figure(7)
    imshow_grid(
        mg1,
        "channel__steepness_index",
        grid_units=("m", "m"),
        var_name="Steepness index",
        cmap="jet"
    )
    title_text = f"$K_{{sp}}$ = {K_sp}; $D$ = {D}; time = {total_time} yr; $dx$ = {dxy} m; concavity = {theta}"
    plt.title(title_text)
    plt.show()


wrapper()
