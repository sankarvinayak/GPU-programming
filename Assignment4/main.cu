#include <iostream>
#include <fstream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/count.h>

#include <time.h>
using namespace std;

// Maximum path length for evacuation routes
// #define MAX_PATH_LENGTH 100
__constant__ int MAX_PATH_LENGTH;
__constant__ int MAX_DROPS;

// Dijkstra's algorithm for finding shortest path of individual populations while running
__device__ int find_shortest_path(
    int start_city,
    const int *row_offsets,
    const int *col_indices,
    const int *lengths,
    int num_cities,
    long long *shelter_capacities,
    int *path)
{
    // Allocate memory for Dijkstra's algorithm data structures
    int *dist = (int *)malloc(num_cities * sizeof(int));
    bool *visited = (bool *)malloc(num_cities * sizeof(bool));
    int *parent = (int *)malloc(num_cities * sizeof(int));

    // Check if memory allocation was successful
    if (dist == nullptr || visited == nullptr || parent == nullptr)
    {
        if (dist)
            free(dist);
        if (visited)
            free(visited);
        if (parent)
            free(parent);
        return -1;
    }

    for (int i = 0; i < num_cities; i++)
    {
        dist[i] = INT_MAX;
        visited[i] = false;
        parent[i] = -1;
    }
    dist[start_city] = 0;

    // Dijkstra's algorithm main loop
    for (int count = 0; count < num_cities; count++)
    {
        int min_dist = INT_MAX;
        int current_city = -1;

        for (int i = 0; i < num_cities; i++)
        {
            if (!visited[i] && dist[i] < min_dist)
            {
                min_dist = dist[i];
                current_city = i;
            }
        }

        if (current_city == -1 || min_dist == INT_MAX)
            break;

        visited[current_city] = true;

        if (shelter_capacities[current_city] > 0)
        {
            int city = current_city;
            int index = 0;

            while (city != -1)
            {
                path[index++] = city;
                city = parent[city];
            }

            for (int i = 0; i < index / 2; i++)
            {
                int temp = path[i];
                path[i] = path[index - i - 1];
                path[index - i - 1] = temp;
            }

            free(dist);
            free(visited);
            free(parent);
            return index; // Return path length
        }

        for (int i = row_offsets[current_city]; i < row_offsets[current_city + 1]; i++)
        {
            int neighbor = col_indices[i];
            int edge_length = lengths[i];
            if (!visited[neighbor] &&
                dist[current_city] != INT_MAX &&
                dist[current_city] + edge_length < dist[neighbor])
            {
                dist[neighbor] = dist[current_city] + edge_length;
                parent[neighbor] = current_city;
            }
        }
    }

    // No path found
    path[0] = -1;

    free(dist);
    free(visited);
    free(parent);
    return -1;
}

// __global__ void path_find_kernel(
//     const int *row_offsets,
//     const int *col_indices,
//     const int *lengths,
//     const int *capacities,
//     const int num_cities,
//     const int *populated_city,
//     // const long long *prime_age_pop,
//     // const long long *elderly_pop,
//     const int num_populated,
//     long long *shelter_capacities,
//     const int max_distance_elderly,
//     int **path_ptrs_dev,
//     int *path_len_dev)
// {
//     int tid = blockIdx.x * blockDim.x + threadIdx.x;
//     if (tid >= num_populated)
//         return;

//     int start_city = populated_city[tid];

//     // Allocate temporary buffer for path
//     int *path_temp = (int *)malloc(num_cities * sizeof(int));
//     if (!path_temp)
//         return;

//     for (int i = 0; i < num_cities; ++i)
//         path_temp[i] = -1;

//     // Find shortest path to a shelter
//     int path_size = find_shortest_path(
//         start_city, row_offsets, col_indices, lengths,
//         num_cities, shelter_capacities, path_temp);

//     // printf("City %d path size: %d\n", start_city, path_size);

//     if (path_size == -1)
//     {
//         free(path_temp);
//         return;
//     }

//     // // Allocate memory for the path

//     // // Update path pointer and length
//     // path_ptrs_dev[tid] = new_path;
//     // path_len_dev[tid] = path_size;
//     int *new_path = path_ptrs_dev[tid];
//     if (path_len_dev[tid] + path_size > MAX_PATH_LENGTH)
//     {
//         // If not enough space, reallocate (increase size)
//         printf("error");
//         return; // Update the local reference
//     }

//     // Copy new path data into the allocated space
//     for (int i = 0; i < path_size; ++i)
//     {
//         new_path[path_len_dev[tid] + i] = path_temp[i];
//     }

//     // Update the path length
//     path_len_dev[tid] += path_size;

//     // Free temporary memory if no longer needed
//     free(path_temp);
//     // for(int i=0;i<path_size;i++)
//     // printf("%d,%d",start_city,path_ptrs_dev[tid][i]);
//     // free(path_temp);
// }

// returns the unique edge id from the CSR representation first occurance as the graph is undirected
__device__ int get_edge_info(
    int city1,
    int city2,
    const int *row_offsets,
    const int *col_indices,
    const int *lengths,
    const int *capacities,
    int *out_length,
    int *out_capacity)
{
    int from_city = min(city1, city2);
    int to_city = max(city1, city2);
    for (int i = row_offsets[from_city]; i < row_offsets[from_city + 1]; ++i)
    {
        if (col_indices[i] == to_city)
        {
            *out_length = lengths[i];
            *out_capacity = capacities[i];
            return i;
        }
    }
    return -1;
}

// due to some reason the target shelter is full now need to find new paths for dropping
__device__ int replan_path(
    int tid,
    const int current_city,
    const int location_index,
    const int city_id,
    const int *row_offsets,
    const int *col_indices,
    const int *lengths,
    const int num_cities,
    long long *shelter_capacities,
    int *path_ptrs_dev,
    int *path_len,
    long long *current_time,
    int *d_finished)
{

    int *path_temp = (int *)malloc(num_cities * sizeof(int));
    if (!path_temp)
        return -1;

    for (int i = 0; i < num_cities; ++i)
        path_temp[i] = -1;

    int new_path_size = find_shortest_path(
        current_city, row_offsets, col_indices, lengths,
        num_cities, shelter_capacities, path_temp);

    if (new_path_size == -1)
    {

        // printf("City %d could not find new path.\n", city_id);
        free(path_temp);

        d_finished[tid] = 1;
        return -1;
    }

    int combined_size = location_index + (new_path_size);
    int *new_path = &path_ptrs_dev[tid * MAX_PATH_LENGTH];
    for (int i = 0; i < new_path_size; i++)
    {
        new_path[location_index + i] = path_temp[i]; // Append the entire new path which may overwrite the existing path to the shelter whch is already full
    }
    path_len[tid] = combined_size;
    free(path_temp);
    return 0;
}

// main kernel in which check if it is available in this time if so check is there any shelter capacity at this point if so drop as much people as possible using locks
// after dropping it may be the case that for some other thread it want to go to the same shelter which is now full so it has to find new paht
// if path is determined now check which edge to take based on the population takes a lock on the edge insted of global barrier give control back to the host
__global__ void simulation_kernel(
    const int *row_offsets,
    const int *col_indices,
    const int *lengths,
    const int *capacities,
    const int num_cities,
    const int *populated_city,
    long long *prime_age_pop,
    long long *elderly_pop,
    const int num_populated,
    long long *shelter_capacities,
    const int max_distance_elderly,
    int *path_ptrs_dev,
    int *path_len,
    long long *current_time,
    int *d_finished,
    int *current_loc,
    int *next_free_time, int *distance_travelled, int *road_availability_time,
    long long *road_claiming_time,
    long long *road_claiming_pop,
    int *road_claiming_tid, int *num_drops,
    long long *drops
    // DeviceVector<Move> *vectors
)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_populated)
        return;
    if (d_finished[tid] == 1)
        return;
    if (*current_time < next_free_time[tid])
        return;
    int location_index = current_loc[tid];
    int city_id = populated_city[tid];
    int current_city = path_ptrs_dev[tid * MAX_PATH_LENGTH + location_index];

    // printf("org city %d Current Time: %lld next free time %d loc_city:%d,index %d %d\n", city_id,*current_time,next_free_time[tid],current_city,location_index,path_len[tid]);

    // while (atomicCAS(&city_lock[current_city], 0, 1))
    //     printf("spinlock");
    if (shelter_capacities[current_city] > 0)
    {

        long long eld = 0;
        long long prime = 0;
       
        long long elderly_to_drop = elderly_pop[tid];
        long long prime_to_drop = prime_age_pop[tid];
        int t = 0;
        
        if (elderly_to_drop > 0)
        {

            long long cap, assumed;
            do
            {
                assumed = shelter_capacities[current_city];
                if (assumed == 0)
                {
                    // Shelter already full, skip drop
                    break;
                }
                long long drop_amount = min(elderly_to_drop, assumed);
                cap = atomicCAS((unsigned long long *)&shelter_capacities[current_city], assumed, assumed - drop_amount);
            } while (cap != assumed);

            if (cap == assumed && cap > 0)
            {
                long long drop_amount = min(elderly_to_drop, cap);

                if (drop_amount == elderly_to_drop)
                {
                    // printf("City %d dropping %lld elderly at %d %d \n", city_id, drop_amount, current_city, next_free_time[tid]);
                    eld = drop_amount;
                    t = 1;
                    elderly_pop[tid] = 0;
                }
                else
                {
                    // printf("City %d dropping %lld elderly at %d (partial)\n", city_id, drop_amount, current_city);
                    elderly_pop[tid] -= drop_amount;

                    // Replan since some elderly are still left
                    int s = replan_path(
                        tid,
                        current_city,
                        location_index,
                        city_id,
                        row_offsets,
                        col_indices,
                        lengths,
                        num_cities,
                        shelter_capacities,
                        path_ptrs_dev,
                        path_len,
                        current_time,
                        d_finished);
                    if (s == -1)
                        current_loc[tid]++;
                }
            }
        }

        if (prime_to_drop > 0)
        {
            long long cap, assumed;
            do
            {
                assumed = shelter_capacities[current_city];
                if (assumed == 0)
                {
                    // Shelter already full, skip drop
                    break;
                }
                long long drop_amount = min(prime_to_drop, assumed);
                cap = atomicCAS((unsigned long long *)&shelter_capacities[current_city], assumed, assumed - drop_amount);
            } while (cap != assumed);

            if (cap == assumed && cap > 0)
            {
                long long drop_amount = min(prime_to_drop, cap);

                if (drop_amount == prime_to_drop)
                {
                    // printf("City %d dropping %lld prime at %d \n", city_id, drop_amount, current_city);
                    prime = drop_amount;
                    prime_age_pop[tid] = 0;
                    d_finished[tid] = 1;
                    current_loc[tid]++;
                    t = 1;
                }
                else
                {
                    // printf("City %d dropping %lld prime at %d (partial)\n", city_id, drop_amount, current_city);
                    prime = drop_amount;
                    prime_age_pop[tid] -= drop_amount;
                    t = 1;

                    int s = replan_path(
                        tid,
                        current_city,
                        location_index,
                        city_id,
                        row_offsets,
                        col_indices,
                        lengths,
                        num_cities,
                        shelter_capacities,
                        path_ptrs_dev,
                        path_len,
                        current_time,
                        d_finished);
                    if (s == -1)
                    {
                        current_loc[tid]++;
                    }
                }
            }
        }
        // Move move = {city_id, current_city, eld, prime};
        // vectors->push_back(move);

        int idx = (tid * MAX_DROPS + num_drops[tid]) * 3;
        drops[idx + 0] = current_city;
        drops[idx + 2] = eld;
        drops[idx + 1] = prime;
        num_drops[tid] += t;

        // city_lock[current_city] = 0;
        // return; // Shelter is either fully occupied or needs replanning
    }
    // city_lock[current_city] = 0;
    if (d_finished[tid] == 1)
        return;
    // If shelter capacity is 0 and thread tries to enter
    if ((shelter_capacities[current_city] <= 0 && location_index == path_len[tid] - 1) || (shelter_capacities[path_len[tid] - 1] == 0))
    {

        // printf("Enter Current Time: %lld", *current_time);
        int s = replan_path(
            tid,
            current_city,
            location_index,
            city_id,
            row_offsets,
            col_indices,
            lengths,
            num_cities,
            shelter_capacities,
            path_ptrs_dev,
            path_len,
            current_time,
            d_finished);
        if (s == -1)
        {
            current_loc[tid]++;
            return;
        }
    }

    // Normal move along path

    int from_city = path_ptrs_dev[tid * MAX_PATH_LENGTH + location_index];
    int to_city = path_ptrs_dev[tid * MAX_PATH_LENGTH + location_index + 1];
    int road_length, road_capacity;
    int road_id = get_edge_info(from_city, to_city, row_offsets, col_indices, lengths, capacities,
                                &road_length, &road_capacity);

    // Get road information
    if (road_availability_time[road_id] <= *current_time)
    {

        int old_time = atomicExch((unsigned long long *)&road_claiming_time[road_id], *current_time);
        if (old_time == 0 || old_time < *current_time)
        {

            atomicExch(&road_claiming_tid[road_id], tid);
            // For long long, we need to use atomicCAS in a loop
            long long old_val, new_val;
            do
            {
                old_val = road_claiming_pop[road_id];
                new_val = elderly_pop[tid] + prime_age_pop[tid];
            } while (atomicCAS((unsigned long long *)&road_claiming_pop[road_id],
                               (unsigned long long)old_val,
                               (unsigned long long)new_val) != (unsigned long long)old_val);

            // can_use_road = true;
        }
        else // if (old_time == *current_time)
        {
            // Road is being claimed this time step, check if we have more population
            // For atomicMax with long long, we need a custom implementation
            long long old_val, new_val;
            do
            {
                old_val = road_claiming_pop[road_id];
                new_val = (elderly_pop[tid] + prime_age_pop[tid] > old_val) ? elderly_pop[tid] + prime_age_pop[tid] : old_val;
            } while (atomicCAS((unsigned long long *)&road_claiming_pop[road_id],
                               (unsigned long long)old_val,
                               (unsigned long long)new_val) != (unsigned long long)old_val);

            if (elderly_pop[tid] + prime_age_pop[tid] > old_val)
            {

                atomicExch(&road_claiming_tid[road_id], tid);
                // can_use_road = true;
            }
            else if (elderly_pop[tid] + prime_age_pop[tid] == old_val && tid < road_claiming_tid[road_id])
            {

                atomicExch(&road_claiming_tid[road_id], tid);
                // can_use_road = true;
            }
        }
    }
}
// based on the lock got from the previous kernel take the edge
__global__ void resolve_dispute(
    const int *row_offsets,
    const int *col_indices,
    const int *lengths,
    const int *capacities,
    const int num_cities,
    const int *populated_city,
    long long *prime_age_pop,
    long long *elderly_pop,
    const int num_populated,
    long long *shelter_capacities,
    const int max_distance_elderly,
    int *path_ptrs_dev,
    // int *path_len,
    long long *current_time,
    int *d_finished,
    int *current_loc,
    int *next_free_time, int *distance_travelled, int *road_availability_time,
    long long *road_claiming_time, 
    long long *road_claiming_pop, 
    int *road_claiming_tid, int *num_drops,
    long long *drops)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_populated)
        return;
    if (d_finished[tid] == 1)
        return;
    if (*current_time < next_free_time[tid])
        return;

    int location_index = current_loc[tid];
    // int city_id = populated_city[tid];
    // int current_city = path_ptrs_dev[tid][location_index];
    int from_city = path_ptrs_dev[tid * MAX_PATH_LENGTH + location_index];
    int to_city = path_ptrs_dev[tid * MAX_PATH_LENGTH + location_index + 1];
    int road_length, road_capacity;
    int road_id = get_edge_info(from_city, to_city, row_offsets, col_indices, lengths, capacities,
                                &road_length, &road_capacity);

    // Check if the road is available
    if (road_availability_time[road_id] > *current_time)
    {
        next_free_time[tid] = road_availability_time[road_id];
        return;
    }

    // if (road_claiming_time[road_id] == *current_time && road_claiming_tid[road_id] != tid) {
    //     printf("tid %d claim by %d claimed at %lld", tid, road_claiming_tid[road_id], road_claiming_time[road_id]);
    // }

    // Check if we got the road
    if (road_claiming_tid[road_id] == tid)
    {
        
        // printf("tid:%d from %d to %d",tid,from_city,to_city);
        if (elderly_pop[tid] > 0 && distance_travelled[tid] + road_length > max_distance_elderly)
        {
            // printf("Dropping elderly city:%d,count %lld\n", city_id, elderly_pop[tid]);

            int idx = (tid * MAX_DROPS + num_drops[tid]) * 3;
            drops[idx + 0] = from_city;
            drops[idx + 2] = elderly_pop[tid];
            drops[idx + 1] = 0;
            elderly_pop[tid] = 0;
            num_drops[tid]++;
        }

        long long population = prime_age_pop[tid] + elderly_pop[tid];
        // int travel_time = ((population + road_capacity - 1) / road_capacity) * (road_length * 12);

        int travel_time = ((population + road_capacity - 1) / road_capacity) * (road_length); // equivalnet to above if time is incrimented by 1 only that is 1 time unit is 12 min

        next_free_time[tid] = *current_time + travel_time;
        // printf("tid %d from %d to %d\n",tid,from_city,to_city);
        // printf("tid: %d, current_time: %lld, travel_time: %d, next_free_time: %d\n",
        //     tid, *current_time, travel_time, next_free_time[tid]);
        current_loc[tid]++;
        distance_travelled[tid] += road_length;

        // Mark road as unavailable 
        atomicMax(&road_availability_time[road_id], next_free_time[tid]);

        // Reset claiming data for next time
        atomicExch((unsigned long long *)&road_claiming_time[road_id], 0);

        // Reset road_claiming_pop 
        long long old_val;
        do
        {
            old_val = road_claiming_pop[road_id];
        } while (atomicCAS((unsigned long long *)&road_claiming_pop[road_id],
                           (unsigned long long)old_val, 0) != (unsigned long long)old_val);

        // Reset claiming thread ID
        atomicExch(&road_claiming_tid[road_id], -1);

        // printf("City %d with pop %lld claimed road %d from %d to %d at time %d\n",
        //        city_id, population, road_id, from_city, to_city, *current_time);
    }
    else
    {
        // We didn't get the road, need to wait
        // printf("tid:%dsetting next free time %d currently claimed by %d",tid,road_availability_time[road_id],road_claiming_tid[road_id]);
        next_free_time[tid] = road_availability_time[road_id];
        // next_free_time[tid] = max(road_availability_time[road_id], (int)*current_time + 1);
    }
}
// timeout is near so drop the remaining population whereever they are right now
__global__ void drop_rest(
    const int *row_offsets,
    const int *col_indices,
    const int *lengths,
    const int *capacities,
    const int num_cities,
    const int *populated_city,
    long long *prime_age_pop,
    long long *elderly_pop,
    const int num_populated,
    long long *shelter_capacities,
    const int max_distance_elderly,
    int *path_ptrs_dev,
    int *path_len,
    long long *current_time,
    int *d_finished,
    int *current_loc,
    int *next_free_time, int *distance_travelled, int *road_avilability_time,
    long long *road_claiming_time,
    long long *road_claiming_pop,
    int *road_claiming_tid, int *num_drops,
    long long *drops
    // DeviceVector<Move> *vectors
)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_populated)
        return;
    if (d_finished[tid] == 1)
        return;
    int location_index = max(current_loc[tid], 0);

    int current_city = path_ptrs_dev[tid * MAX_PATH_LENGTH + location_index];
    // int city_id = populated_city[tid];
    long long elderly_to_drop = elderly_pop[tid];
    long long prime_to_drop = prime_age_pop[tid];
    // printf("%d free at %d\n", city_id, next_free_time[tid]);
    int t = 0;
    if (prime_to_drop > 0)
    {
        // printf("City %d dropping %lld prime at %d\n", city_id, prime_to_drop, current_city);
        t = 1;
    }

    if (elderly_to_drop > 0)
    {
        // printf("City %d dropping %lld elderly at %d\n", city_id, elderly_to_drop, current_city);
        t = 1;
    }
    int idx = (tid * MAX_DROPS + num_drops[tid]) * 3;
    drops[idx + 0] = current_city;
    drops[idx + 2] = elderly_to_drop;
    drops[idx + 1] = prime_to_drop;
    num_drops[tid] += t;
}
// for finding shortest path initialize the distances
__global__ void init_distances(
    long long *distance_to_shelter,
    long long *next_distance_to_shelter,
    const long long *shelter_capacities,
    int num_cities)
{
    int city = blockIdx.x * blockDim.x + threadIdx.x;
    if (city >= num_cities)
        return;

    if (shelter_capacities[city] > 0)
    {
        
        distance_to_shelter[city] = 0;
        next_distance_to_shelter[city] = 0;
    }
    else
    {
        
        distance_to_shelter[city] = LLONG_MAX;
        next_distance_to_shelter[city] = LLONG_MAX;
    }
}

__global__ void relax_distances(
    const int *row_offsets,
    const int *col_indices,
    const int *lengths,
    const long long *shelter_capacities,
    long long *distance_to_shelter,
    long long *next_distance_to_shelter,
    bool *updated,
    int num_cities)
{
    int city = blockIdx.x * blockDim.x + threadIdx.x;
    if (city >= num_cities)
        return;

    long long current_dist = distance_to_shelter[city];
    long long new_dist = current_dist;

    
    if (shelter_capacities[city] > 0)
    {
        new_dist = 0;
    }
    else
    {
        
        for (int i = row_offsets[city]; i < row_offsets[city + 1]; i++)
        {
            int neighbor = col_indices[i];
            int edge_len = lengths[i];

            
            if (distance_to_shelter[neighbor] != LLONG_MAX)
            {
                long long alt = distance_to_shelter[neighbor] + edge_len;
                if (alt < new_dist)
                {
                    new_dist = alt;
                }
            }
        }
    }

    next_distance_to_shelter[city] = new_dist;
    if (new_dist < current_dist)
    {
        updated[city] = true;
    }
}

__global__ void reconstruct_path(
    const int *row_offsets,
    const int *col_indices,
    const int *lengths,
    const long long *distance_to_shelter,
    const long long *shelter_capacities,
    int *path_ptrs_dev,
    int *path_len_dev, int *populated_city,
    int num_cities,
    int num_populated)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_populated)
        return;
    int city_id = populated_city[tid];
    
    if (shelter_capacities[city_id] > 0)
    {
        path_ptrs_dev[tid * MAX_PATH_LENGTH + 0] = city_id;
        path_len_dev[tid] = 1;
        return;
    }

    
    if (distance_to_shelter[tid] == LLONG_MAX)
    {
        path_len_dev[city_id] = 0;
        return;
    }

    int current = city_id;
    int path_len = 0;
    int max_hops = num_cities; 

    
    while (path_len < max_hops)
    {
        
        path_ptrs_dev[tid * MAX_PATH_LENGTH + path_len++] = current;

        
        if (shelter_capacities[current] > 0)
            break;

       
        int next_city = -1;
        long long current_dist = distance_to_shelter[current];

        for (int i = row_offsets[current]; i < row_offsets[current + 1]; i++)
        {
            int neighbor = col_indices[i];
            int edge_len = lengths[i];

           
            if (distance_to_shelter[neighbor] + edge_len == current_dist)
            {
                next_city = neighbor;
                break; 
            }
        }

        if (next_city == -1)
        {
            
            path_len = 0;
            break;
        }

        current = next_city;
    }

    path_len_dev[tid] = path_len;
}
struct IsValidNextFree
{
    long long simulation_time;

    __host__ __device__
    IsValidNextFree(long long sim_time) : simulation_time(sim_time) {}

    __host__ __device__ bool operator()(const thrust::tuple<long long, int> &t) const
    {
        return thrust::get<1>(t) == 0 && thrust::get<0>(t) > simulation_time;
    }
};
__global__ void fallback_kernel(
    const int *row_offsets,
    const int *col_indices,
    const int *lengths,
    const int *capacities,
    int num_cities,
    int num_edges,
    int *fallback_path,
    long long *fallback_drops,
    int *path_size,
    int *num_drops,
    int num_populated_cities,
    const long long *shelter_capacities,
    const int *populated_city,
    const long long *prime_age_pop,
    const long long *elderly_pop)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_populated_cities)
        return;
    int city_id = populated_city[tid];
    long long total_pop = prime_age_pop[tid] + elderly_pop[tid];
    long long cap = shelter_capacities[city_id];
    if (cap > total_pop || cap == 0)
    {
        path_size[tid] = 1;
        num_drops[tid] = 1;
        fallback_path[tid] = city_id;
        fallback_drops[tid * 3 + 0] = city_id;
        fallback_drops[tid * 3 + 1] = prime_age_pop[tid];
        fallback_drops[tid * 3 + 2] = elderly_pop[tid];
    }
    else
    {
        int neighbor = -1;
        for (int i = row_offsets[city_id]; i < row_offsets[city_id + 1]; ++i)
        {
            neighbor = col_indices[i];
            break;
        }
        if (neighbor == -1)
            neighbor = city_id; 

        path_size[tid] = 2;
        num_drops[tid] = 2;
        fallback_path[tid] = city_id;
        fallback_path[tid + num_populated_cities] = neighbor;

        long long drop_at_city = cap;

        long long prime_drop = min(drop_at_city, prime_age_pop[tid]);
        long long elderly_drop = drop_at_city - prime_drop;

        fallback_drops[tid * 3 + 0] = city_id;
        fallback_drops[tid * 3 + 1] = prime_drop;
        fallback_drops[tid * 3 + 2] = elderly_drop;

        long long prime_rem = prime_age_pop[tid] - prime_drop;
        long long elderly_rem = elderly_pop[tid] - elderly_drop;
        fallback_drops[(tid + num_populated_cities) * 3 + 0] = neighbor;
        fallback_drops[(tid + num_populated_cities) * 3 + 1] = prime_rem;
        fallback_drops[(tid + num_populated_cities) * 3 + 2] = elderly_rem;
    }
}

void backup(
    const char *output_filename,
    long long num_populated_cities,
    int h_max_path_length,
    int h_max_drops,
    const int *d_row_offsets,
    const int *d_col_indices,
    const int *d_lengths,
    const int *d_capacities,
    int num_cities,
    int num_edges,
    const int *h_populated_city,
    const long long *h_prime_age_pop,
    const long long *h_elderly_pop,
    const long long *d_shelter_capacities,
    const int *d_populated_city,
    const long long *d_prime_age_pop,
    const long long *d_elderly_pop,
    const long long *h_shelter_capacities 
)
{
    std::ofstream outfile(output_filename);
    if (!outfile)
    {
        std::cerr << "Error: Cannot open file " << output_filename << "\n";
        return;
    }

    int *fallback_path = nullptr;
    long long *fallback_drops = nullptr;
    int *path_size_dev = nullptr;
    int *num_drops_dev = nullptr;
    cudaError_t err1 = cudaMalloc(&fallback_path, 2 * num_populated_cities * sizeof(int));
    cudaError_t err2 = cudaMalloc(&fallback_drops, 2 * 3 * num_populated_cities * sizeof(long long));
    cudaError_t err3 = cudaMalloc(&path_size_dev, num_populated_cities * sizeof(int));
    cudaError_t err4 = cudaMalloc(&num_drops_dev, num_populated_cities * sizeof(int));
    if (err1 != cudaSuccess || err2 != cudaSuccess || err3 != cudaSuccess || err4 != cudaSuccess)
    {
        if (fallback_path)
            cudaFree(fallback_path);
        if (fallback_drops)
            cudaFree(fallback_drops);
        long long *path_size = new long long[num_populated_cities];
        int **paths = new int *[num_populated_cities];
        long long *num_drops = new long long[num_populated_cities];
        long long ***drops = new long long **[num_populated_cities];
        for (int i = 0; i < num_populated_cities; ++i)
        {
            path_size[i] = 1;
            paths[i] = new int[1];
            paths[i][0] = h_populated_city[i];
            num_drops[i] = 1;
            drops[i] = new long long *[1];
            drops[i][0] = new long long[3];
            drops[i][0][0] = h_populated_city[i];
            drops[i][0][1] = h_prime_age_pop[i];
            drops[i][0][2] = h_elderly_pop[i];
        }
        for (long long i = 0; i < num_populated_cities; i++)
        {
            for (long long j = 0; j < path_size[i]; j++)
            {
                outfile << paths[i][j] << " ";
            }
            outfile << "\n";
        }
        for (long long i = 0; i < num_populated_cities; i++)
        {
            for (long long j = 0; j < num_drops[i]; j++)
            {
                for (int k = 0; k < 3; k++)
                {
                    outfile << drops[i][j][k] << " ";
                }
            }
            outfile << "\n";
        }
        for (int i = 0; i < num_populated_cities; ++i)
        {
            delete[] paths[i];
            delete[] drops[i][0];
            delete[] drops[i];
        }
        delete[] path_size;
        delete[] paths;
        delete[] num_drops;
        delete[] drops;
        outfile.close();
        return;
    }
    cudaMemset(fallback_path, 0, 2 * num_populated_cities * sizeof(int));
    cudaMemset(fallback_drops, 0, 2 * 3 * num_populated_cities * sizeof(long long));
    cudaMemset(path_size_dev, 0, num_populated_cities * sizeof(int));
    cudaMemset(num_drops_dev, 0, num_populated_cities * sizeof(int));

    long long *path_size = new long long[num_populated_cities];
    int **paths = new int *[num_populated_cities];
    long long *num_drops = new long long[num_populated_cities];
    long long ***drops = new long long **[num_populated_cities];
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_populated_cities + threadsPerBlock - 1) / threadsPerBlock;
    fallback_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_row_offsets, d_col_indices, d_lengths, d_capacities, num_cities, num_edges,
        fallback_path, fallback_drops, path_size_dev, num_drops_dev, num_populated_cities,
        d_shelter_capacities, d_populated_city, d_prime_age_pop, d_elderly_pop);
    cudaDeviceSynchronize();

    int *path_size_tmp = new int[num_populated_cities];
    int *num_drops_tmp = new int[num_populated_cities];
    int *fallback_path_host = new int[2 * num_populated_cities];
    long long *fallback_drops_host = new long long[2 * 3 * num_populated_cities];
    cudaMemcpy(path_size_tmp, path_size_dev, num_populated_cities * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(num_drops_tmp, num_drops_dev, num_populated_cities * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(fallback_path_host, fallback_path, 2 * num_populated_cities * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(fallback_drops_host, fallback_drops, 2 * 3 * num_populated_cities * sizeof(long long), cudaMemcpyDeviceToHost);

    for (int i = 0; i < num_populated_cities; ++i)
    {
        path_size[i] = path_size_tmp[i];
        paths[i] = new int[path_size[i]];
        for (int j = 0; j < path_size[i]; ++j)
        {
            paths[i][j] = fallback_path_host[i + j];
        }
        num_drops[i] = num_drops_tmp[i];
        drops[i] = new long long *[num_drops[i]];
        for (int j = 0; j < num_drops[i]; ++j)
        {
            drops[i][j] = new long long[3];
            for (int k = 0; k < 3; ++k)
            {
                drops[i][j][k] = fallback_drops_host[(i * 3) + k];
            }
        }
    }

    
    for (long long i = 0; i < num_populated_cities; i++)
    {
        long long currentPathSize = path_size[i];
        for (long long j = 0; j < currentPathSize; j++)
        {
            outfile << paths[i][j] << " ";
        }
        outfile << "\n";
    }

    for (long long i = 0; i < num_populated_cities; i++)
    {
        long long currentDropSize = num_drops[i];
        for (long long j = 0; j < currentDropSize; j++)
        {
            for (int k = 0; k < 3; k++)
            {
                outfile << drops[i][j][k] << " ";
            }
        }
        outfile << "\n";
    }

    cudaFree(fallback_path);
    cudaFree(fallback_drops);
    cudaFree(path_size_dev);
    cudaFree(num_drops_dev);
    delete[] path_size_tmp;
    delete[] num_drops_tmp;
    delete[] fallback_path_host;
    delete[] fallback_drops_host;

    outfile.close();
}
int main(int argc, char *argv[])
{

    time_t start_time = time(NULL);
    // Validate command line arguments
    if (argc < 3)
    {
        std::cerr << "Usage: " << argv[0] << " <input_file> <output_file>\n";
        return 1;
    }

    // Open input file
    std::ifstream infile(argv[1]);
    if (!infile)
    {
        std::cerr << "Error: Cannot open file " << argv[1] << "\n";
        return 1;
    }

    // Read number of cities and roads
    long long num_cities, num_roads;
    infile >> num_cities >> num_roads;

    // Create vectors to store edge data
    thrust::host_vector<int> h_sources;
    thrust::host_vector<int> h_destinations;
    thrust::host_vector<int> h_lengths;
    thrust::host_vector<int> h_capacities;

    // Reserve space for bidirectional edges
    h_sources.reserve(num_roads * 2);
    h_destinations.reserve(num_roads * 2);
    h_lengths.reserve(num_roads * 2);
    h_capacities.reserve(num_roads * 2);

    // Read road data
    for (int i = 0; i < num_roads; i++)
    {
        int u, v, length, capacity;
        infile >> u >> v >> length >> capacity;

        // Add edge in both directions (bidirectional graph)
        h_sources.push_back(u);
        h_destinations.push_back(v);
        h_lengths.push_back(length);
        h_capacities.push_back(capacity);

        h_sources.push_back(v);
        h_destinations.push_back(u);
        h_lengths.push_back(length);
        h_capacities.push_back(capacity);
    }

    // Read shelter data
    int num_shelters;
    infile >> num_shelters;

    thrust::host_vector<long long> h_shelter_capacities(num_cities, 0);
    for (int i = 0; i < num_shelters; i++)
    {
        int city_id;
        long long capacity;
        infile >> city_id >> capacity;
        h_shelter_capacities[city_id] = capacity;
    }

    // Read populated cities data
    int num_populated_cities;
    infile >> num_populated_cities;

    thrust::host_vector<int> h_populated_city(num_populated_cities);
    thrust::host_vector<long long> h_prime_age_pop(num_populated_cities);
    thrust::host_vector<long long> h_elderly_pop(num_populated_cities);

    for (int i = 0; i < num_populated_cities; i++)
    {
        infile >> h_populated_city[i] >> h_prime_age_pop[i] >> h_elderly_pop[i];
    }

    // Read max distance for elderly
    int max_distance_elderly;
    infile >> max_distance_elderly;
    infile.close();

    // Total number of edges (after making graph bidirectional)
    const int num_edges = h_sources.size();

    // Copy edge data to device
    thrust::device_vector<int> d_sources(h_sources);
    thrust::device_vector<int> d_destinations(h_destinations);
    thrust::device_vector<int> d_lengths(h_lengths);
    thrust::device_vector<int> d_capacities(h_capacities);

    // Create index array for sorting
    thrust::device_vector<int> d_indices(num_edges);
    thrust::sequence(d_indices.begin(), d_indices.end());

    // Sort edges by source vertex (for CSR format)
    thrust::sort_by_key(
        d_sources.begin(),
        d_sources.end(),
        thrust::make_zip_iterator(thrust::make_tuple(
            d_destinations.begin(),
            d_lengths.begin(),
            d_capacities.begin(),
            d_indices.begin())));

    // Create CSR row offsets

    thrust::host_vector<int> row_counts(num_cities + 1, 0);
    thrust::host_vector<int> h_sources_sorted = d_sources;

    // Count edges per source vertex
    for (int i = 0; i < num_edges; ++i)
    {
        row_counts[h_sources_sorted[i]]++;
    }

    // Convert counts to CSR offsets
    thrust::device_vector<int> d_row_counts(row_counts);
    thrust::device_vector<int> d_row_offsets(num_cities + 1);
    thrust::exclusive_scan(d_row_counts.begin(), d_row_counts.end(), d_row_offsets.begin());

    // Copy city data to device
    thrust::device_vector<long long> d_shelter_capacities = h_shelter_capacities;
    thrust::device_vector<int> d_populated_city = h_populated_city;
    thrust::device_vector<long long> d_prime_age_pop = h_prime_age_pop;
    thrust::device_vector<long long> d_elderly_pop = h_elderly_pop;

    // Allocate memory for evacuation paths

    // define max limits to limit the memory consumption
    cudaError_t err;
    int h_max_path_length = 2 * num_cities;
    int h_max_drops = num_cities;

    //  path preallocate on gpu
    // int **path_ptrs_dev;
    // cudaMalloc(&path_ptrs_dev, num_populated_cities * sizeof(int *));
    // // Step 2: Allocate memory for each city's path array and store the pointers in path_ptrs_dev
    // for (int i = 0; i < num_populated_cities; i++)
    // {
    //     int *path_array_dev;
    //     cudaMalloc(&path_array_dev, h_max_path_length * num_cities * sizeof(int)); // Allocate array of size 3 * num_cities for each city
    //     // Store the pointer to this array in the path_ptrs_dev array
    //     cudaMemcpy(&path_ptrs_dev[i], &path_array_dev, sizeof(int *), cudaMemcpyHostToDevice);
    // }
    int *path_array_dev;
    err = cudaMalloc(&path_array_dev, num_populated_cities * h_max_path_length * num_cities * sizeof(int));
    if (err != cudaSuccess)
    {
        backup(argv[2], num_populated_cities, h_max_path_length, h_max_drops,
               thrust::raw_pointer_cast(d_row_offsets.data()),
               thrust::raw_pointer_cast(d_destinations.data()),
               thrust::raw_pointer_cast(d_lengths.data()),
               thrust::raw_pointer_cast(d_capacities.data()),
               num_cities, num_edges,
               thrust::raw_pointer_cast(d_populated_city.data()),
               thrust::raw_pointer_cast(d_prime_age_pop.data()),
               thrust::raw_pointer_cast(d_elderly_pop.data()),
               thrust::raw_pointer_cast(d_shelter_capacities.data()),
               thrust::raw_pointer_cast(d_populated_city.data()),
               thrust::raw_pointer_cast(d_prime_age_pop.data()),
               thrust::raw_pointer_cast(d_elderly_pop.data()),
               thrust::raw_pointer_cast(h_shelter_capacities.data()));
        return 1;
    }
    // int *path_len_dev;
    // cudaMalloc(&path_len_dev, num_populated_cities * sizeof(int));
    // cudaMemset(path_len_dev, 0, sizeof(int) * num_populated_cities);
    thrust::device_vector<int> path_len_dev(num_populated_cities, 0);

    thrust::device_vector<int> num_drops_gpu(num_populated_cities, 0);

    // Host
    // long long ***d_drops;
    // cudaMalloc(&d_drops, num_populated_cities * sizeof(long long **));
    // long long ***h_drops = new long long **[num_populated_cities];
    // for (int i = 0; i < num_populated_cities; ++i)
    // {
    //     cudaMalloc(&h_drops[i], h_max_drops * sizeof(long long *));
    //     long long **h_city_drops = new long long *[h_max_drops];
    //     for (int j = 0; j < h_max_drops; ++j)
    //     {
    //         cudaMalloc(&h_city_drops[j], 3 * sizeof(long long));
    //     }
    //     cudaMemcpy(h_drops[i], h_city_drops, h_max_drops * sizeof(long long *), cudaMemcpyHostToDevice);
    //     delete[] h_city_drops;
    // }
    // cudaMemcpy(d_drops, h_drops, num_populated_cities * sizeof(long long **), cudaMemcpyHostToDevice);
    // delete[] h_drops;
    // Total number of long longs needed = num_populated_cities * h_max_drops * 3
    long long *d_drops;
    err = cudaMalloc(&d_drops, num_populated_cities * h_max_drops * 3 * sizeof(long long));
    if (err != cudaSuccess)
    {
        cudaFree(path_array_dev);
        h_max_path_length = num_cities;
        h_max_drops = num_cities / 2;
        err = cudaMalloc(&path_array_dev, num_populated_cities * h_max_path_length * num_cities * sizeof(int));
        if (err != cudaSuccess)
        {
            backup(argv[2], num_populated_cities, h_max_path_length, h_max_drops,
                   thrust::raw_pointer_cast(d_row_offsets.data()),
                   thrust::raw_pointer_cast(d_destinations.data()),
                   thrust::raw_pointer_cast(d_lengths.data()),
                   thrust::raw_pointer_cast(d_capacities.data()),
                   num_cities, num_edges,
                   thrust::raw_pointer_cast(d_populated_city.data()),
                   thrust::raw_pointer_cast(d_prime_age_pop.data()),
                   thrust::raw_pointer_cast(d_elderly_pop.data()),
                   thrust::raw_pointer_cast(d_shelter_capacities.data()),
                   thrust::raw_pointer_cast(d_populated_city.data()),
                   thrust::raw_pointer_cast(d_prime_age_pop.data()),
                   thrust::raw_pointer_cast(d_elderly_pop.data()),
                   thrust::raw_pointer_cast(h_shelter_capacities.data()));
            return 1;
        }
        err = cudaMalloc(&d_drops, num_populated_cities * h_max_drops * 3 * sizeof(long long));
        if (err != cudaSuccess)
        {
            backup(argv[2], num_populated_cities, h_max_path_length, h_max_drops,
                   thrust::raw_pointer_cast(d_row_offsets.data()),
                   thrust::raw_pointer_cast(d_destinations.data()),
                   thrust::raw_pointer_cast(d_lengths.data()),
                   thrust::raw_pointer_cast(d_capacities.data()),
                   num_cities, num_edges,
                   thrust::raw_pointer_cast(d_populated_city.data()),
                   thrust::raw_pointer_cast(d_prime_age_pop.data()),
                   thrust::raw_pointer_cast(d_elderly_pop.data()),
                   thrust::raw_pointer_cast(d_shelter_capacities.data()),
                   thrust::raw_pointer_cast(d_populated_city.data()),
                   thrust::raw_pointer_cast(d_prime_age_pop.data()),
                   thrust::raw_pointer_cast(d_elderly_pop.data()),
                   thrust::raw_pointer_cast(h_shelter_capacities.data()));
            return 1;
        }
    }

    cudaMemcpyToSymbol(MAX_PATH_LENGTH, &h_max_path_length, sizeof(int));
    cudaMemcpyToSymbol(MAX_DROPS, &h_max_drops, sizeof(int));
    // Allocate simulation variables
    long long *d_timer;
    err = cudaMallocManaged(&d_timer, sizeof(long));
    if (err != cudaSuccess)
    {
        backup(argv[2], num_populated_cities, h_max_path_length, h_max_drops,
               thrust::raw_pointer_cast(d_row_offsets.data()),
               thrust::raw_pointer_cast(d_destinations.data()),
               thrust::raw_pointer_cast(d_lengths.data()),
               thrust::raw_pointer_cast(d_capacities.data()),
               num_cities, num_edges,
               thrust::raw_pointer_cast(d_populated_city.data()),
               thrust::raw_pointer_cast(d_prime_age_pop.data()),
               thrust::raw_pointer_cast(d_elderly_pop.data()),
               thrust::raw_pointer_cast(d_shelter_capacities.data()),
               thrust::raw_pointer_cast(d_populated_city.data()),
               thrust::raw_pointer_cast(d_prime_age_pop.data()),
               thrust::raw_pointer_cast(d_elderly_pop.data()),
               thrust::raw_pointer_cast(h_shelter_capacities.data()));
        return 1;
    }
    *d_timer = 0;

    // int *d_unfinished;
    // cudaMalloc(&d_unfinished, sizeof(int));
    int h_finished = 0;
    // cudaMemcpy(d_unfinished, &h_unfinished, sizeof(int), cudaMemcpyHostToDevice);
    thrust::device_vector<int> d_finished(num_populated_cities, 0);

    // Track current location and next movement time for each city
    thrust::device_vector<int> current_loc(num_populated_cities, 0);
    thrust::device_vector<int> next_free(num_populated_cities, 0);
    thrust::device_vector<int> distance_travelled(num_populated_cities, 0);
    thrust::device_vector<int> road_avilability_time(num_roads, 0);
    // Simulation loop - continue until all evacuees reach shelters
    long long simulation_time = 0;
    long long *d_road_claiming_time;
    long long *d_road_claiming_pop;
    int *d_road_claiming_tid;
    err = cudaMalloc((void **)&d_road_claiming_time, num_roads * sizeof(long long));
    if (err != cudaSuccess)
    {
        backup(argv[2], num_populated_cities, h_max_path_length, h_max_drops,
               thrust::raw_pointer_cast(d_row_offsets.data()),
               thrust::raw_pointer_cast(d_destinations.data()),
               thrust::raw_pointer_cast(d_lengths.data()),
               thrust::raw_pointer_cast(d_capacities.data()),
               num_cities, num_edges,
               thrust::raw_pointer_cast(d_populated_city.data()),
               thrust::raw_pointer_cast(d_prime_age_pop.data()),
               thrust::raw_pointer_cast(d_elderly_pop.data()),
               thrust::raw_pointer_cast(d_shelter_capacities.data()),
               thrust::raw_pointer_cast(d_populated_city.data()),
               thrust::raw_pointer_cast(d_prime_age_pop.data()),
               thrust::raw_pointer_cast(d_elderly_pop.data()),
               thrust::raw_pointer_cast(h_shelter_capacities.data()));
        return 1;
    }
    err = cudaMalloc((void **)&d_road_claiming_pop, num_roads * sizeof(long long));
    if (err != cudaSuccess)
    {
        backup(argv[2], num_populated_cities, h_max_path_length, h_max_drops,
               thrust::raw_pointer_cast(d_row_offsets.data()),
               thrust::raw_pointer_cast(d_destinations.data()),
               thrust::raw_pointer_cast(d_lengths.data()),
               thrust::raw_pointer_cast(d_capacities.data()),
               num_cities, num_edges,
               thrust::raw_pointer_cast(d_populated_city.data()),
               thrust::raw_pointer_cast(d_prime_age_pop.data()),
               thrust::raw_pointer_cast(d_elderly_pop.data()),
               thrust::raw_pointer_cast(d_shelter_capacities.data()),
               thrust::raw_pointer_cast(d_populated_city.data()),
               thrust::raw_pointer_cast(d_prime_age_pop.data()),
               thrust::raw_pointer_cast(d_elderly_pop.data()),
               thrust::raw_pointer_cast(h_shelter_capacities.data()));
        return 1;
    }
    err = cudaMalloc((void **)&d_road_claiming_tid, num_roads * sizeof(int));
    if (err != cudaSuccess)
    {
        backup(argv[2], num_populated_cities, h_max_path_length, h_max_drops,
               thrust::raw_pointer_cast(d_row_offsets.data()),
               thrust::raw_pointer_cast(d_destinations.data()),
               thrust::raw_pointer_cast(d_lengths.data()),
               thrust::raw_pointer_cast(d_capacities.data()),
               num_cities, num_edges,
               thrust::raw_pointer_cast(d_populated_city.data()),
               thrust::raw_pointer_cast(d_prime_age_pop.data()),
               thrust::raw_pointer_cast(d_elderly_pop.data()),
               thrust::raw_pointer_cast(d_shelter_capacities.data()),
               thrust::raw_pointer_cast(d_populated_city.data()),
               thrust::raw_pointer_cast(d_prime_age_pop.data()),
               thrust::raw_pointer_cast(d_elderly_pop.data()),
               thrust::raw_pointer_cast(h_shelter_capacities.data()));
        return 1;
    }

    // Initialize arrays
    cudaMemset(d_road_claiming_time, 0, num_roads * sizeof(long long));
    cudaMemset(d_road_claiming_pop, 0, num_roads * sizeof(long long));
    cudaMemset(d_road_claiming_tid, -1, num_roads * sizeof(int));

    // Configure kernel launch parameters
    // path_find_kernel<<<blocksPerGrid, threadsPerBlock>>>(
    //     thrust::raw_pointer_cast(d_row_offsets.data()),
    //     thrust::raw_pointer_cast(d_destinations.data()),
    //     thrust::raw_pointer_cast(d_lengths.data()),
    //     thrust::raw_pointer_cast(d_capacities.data()),
    //     num_cities,
    //     thrust::raw_pointer_cast(d_populated_city.data()),
    //     // thrust::raw_pointer_cast(d_prime_age_pop.data()),
    //     // thrust::raw_pointer_cast(d_elderly_pop.data()),
    //     num_populated_cities,
    //     thrust::raw_pointer_cast(d_shelter_capacities.data()),
    //     max_distance_elderly,
    //     path_ptrs_dev,
    //     thrust::raw_pointer_cast(path_len_dev.data()));

    // cudaDeviceSynchronize();
    bool *d_updated;
    long long *d_distance_to_shelter, *d_next_distance_to_shelter;

    // Allocate memory
    err = cudaMalloc(&d_updated, num_cities * sizeof(bool));
    if (err != cudaSuccess)
    {
        backup(argv[2], num_populated_cities, h_max_path_length, h_max_drops,
               thrust::raw_pointer_cast(d_row_offsets.data()),
               thrust::raw_pointer_cast(d_destinations.data()),
               thrust::raw_pointer_cast(d_lengths.data()),
               thrust::raw_pointer_cast(d_capacities.data()),
               num_cities, num_edges,
               thrust::raw_pointer_cast(d_populated_city.data()),
               thrust::raw_pointer_cast(d_prime_age_pop.data()),
               thrust::raw_pointer_cast(d_elderly_pop.data()),
               thrust::raw_pointer_cast(d_shelter_capacities.data()),
               thrust::raw_pointer_cast(d_populated_city.data()),
               thrust::raw_pointer_cast(d_prime_age_pop.data()),
               thrust::raw_pointer_cast(d_elderly_pop.data()),
               thrust::raw_pointer_cast(h_shelter_capacities.data()));
        return 1;
    }
    err = cudaMalloc(&d_distance_to_shelter, sizeof(long long) * num_cities);
    if (err != cudaSuccess)
    {
        backup(argv[2], num_populated_cities, h_max_path_length, h_max_drops,
               thrust::raw_pointer_cast(d_row_offsets.data()),
               thrust::raw_pointer_cast(d_destinations.data()),
               thrust::raw_pointer_cast(d_lengths.data()),
               thrust::raw_pointer_cast(d_capacities.data()),
               num_cities, num_edges,
               thrust::raw_pointer_cast(d_populated_city.data()),
               thrust::raw_pointer_cast(d_prime_age_pop.data()),
               thrust::raw_pointer_cast(d_elderly_pop.data()),
               thrust::raw_pointer_cast(d_shelter_capacities.data()),
               thrust::raw_pointer_cast(d_populated_city.data()),
               thrust::raw_pointer_cast(d_prime_age_pop.data()),
               thrust::raw_pointer_cast(d_elderly_pop.data()),
               thrust::raw_pointer_cast(h_shelter_capacities.data()));
        return 1;
    }
    err = cudaMalloc(&d_next_distance_to_shelter, sizeof(long long) * num_cities);
    if (err != cudaSuccess)
    {
        backup(argv[2], num_populated_cities, h_max_path_length, h_max_drops,
               thrust::raw_pointer_cast(d_row_offsets.data()),
               thrust::raw_pointer_cast(d_destinations.data()),
               thrust::raw_pointer_cast(d_lengths.data()),
               thrust::raw_pointer_cast(d_capacities.data()),
               num_cities, num_edges,
               thrust::raw_pointer_cast(d_populated_city.data()),
               thrust::raw_pointer_cast(d_prime_age_pop.data()),
               thrust::raw_pointer_cast(d_elderly_pop.data()),
               thrust::raw_pointer_cast(d_shelter_capacities.data()),
               thrust::raw_pointer_cast(d_populated_city.data()),
               thrust::raw_pointer_cast(d_prime_age_pop.data()),
               thrust::raw_pointer_cast(d_elderly_pop.data()),
               thrust::raw_pointer_cast(h_shelter_capacities.data()));
        return 1;
    }
    // h_updated = new bool[num_cities];

    // Initialize distance arrays
    init_distances<<<(num_cities + 1023) / 1024, 1024>>>(
        d_distance_to_shelter,
        d_next_distance_to_shelter,
        thrust::raw_pointer_cast(d_shelter_capacities.data()),
        num_cities);

    int threads_per_block = 1024;
    int blocks = (num_cities + threads_per_block - 1) / threads_per_block;

    thrust::device_ptr<bool> d_ptr(d_updated);
    // Bellman-Ford iterations to find shortest paths
    for (int iter = 0; iter < num_cities; ++iter)
    {
        cudaMemset(d_updated, 0, num_cities * sizeof(bool));

        relax_distances<<<blocks, threads_per_block>>>(
            thrust::raw_pointer_cast(d_row_offsets.data()),
            thrust::raw_pointer_cast(d_destinations.data()),
            thrust::raw_pointer_cast(d_lengths.data()),
            thrust::raw_pointer_cast(d_shelter_capacities.data()),
            d_distance_to_shelter,
            d_next_distance_to_shelter,
            d_updated,
            num_cities);
        cudaDeviceSynchronize();
        // Copy updated flags back to host
        // cudaMemcpy(h_updated, d_updated, num_cities * sizeof(bool), cudaMemcpyDeviceToHost);

        bool any_updated = thrust::reduce(
            d_ptr,
            d_ptr + num_cities,
            false,
            thrust::logical_or<bool>());

        if (!any_updated)
            break;

        std::swap(d_distance_to_shelter, d_next_distance_to_shelter);
    }

    int threadsPerBlock = 128;
    int blocksPerGrid = (num_populated_cities + threadsPerBlock - 1) / threadsPerBlock;

    // Now reconstruct the paths from each populated city to the nearest shelter
    reconstruct_path<<<blocks, threads_per_block>>>(
        thrust::raw_pointer_cast(d_row_offsets.data()),
        thrust::raw_pointer_cast(d_destinations.data()),
        thrust::raw_pointer_cast(d_lengths.data()),
        d_distance_to_shelter,
        thrust::raw_pointer_cast(d_shelter_capacities.data()),
        path_array_dev,
        thrust::raw_pointer_cast(path_len_dev.data()),
        thrust::raw_pointer_cast(d_populated_city.data()),
        num_cities, num_populated_cities);

    // delete[] h_updated;
    cudaFree(d_updated);
    cudaFree(d_distance_to_shelter);
    cudaFree(d_next_distance_to_shelter);
    thrust::device_vector<long long> valid_times(num_populated_cities);
    auto zip_begin = thrust::make_zip_iterator(thrust::make_tuple(next_free.begin(), d_finished.begin()));
    auto zip_end = zip_begin + num_populated_cities;
    while (true)
    {
        *d_timer = simulation_time;

        // Run simulation step
        simulation_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            thrust::raw_pointer_cast(d_row_offsets.data()),
            thrust::raw_pointer_cast(d_destinations.data()),
            thrust::raw_pointer_cast(d_lengths.data()),
            thrust::raw_pointer_cast(d_capacities.data()),
            num_cities,
            thrust::raw_pointer_cast(d_populated_city.data()),
            thrust::raw_pointer_cast(d_prime_age_pop.data()),
            thrust::raw_pointer_cast(d_elderly_pop.data()),
            num_populated_cities,
            thrust::raw_pointer_cast(d_shelter_capacities.data()),
            max_distance_elderly,
            path_array_dev,
            thrust::raw_pointer_cast(path_len_dev.data()),
            d_timer,
            thrust::raw_pointer_cast(d_finished.data()),
            thrust::raw_pointer_cast(current_loc.data()),
            thrust::raw_pointer_cast(next_free.data()), thrust::raw_pointer_cast(distance_travelled.data()), thrust::raw_pointer_cast(road_avilability_time.data()),
            d_road_claiming_time,
            d_road_claiming_pop,
            d_road_claiming_tid,
            thrust::raw_pointer_cast(num_drops_gpu.data()),
            d_drops
            // , device_vectors
        );
        // resolve the dispute aka edge contentions
        resolve_dispute<<<blocksPerGrid, threadsPerBlock>>>(
            thrust::raw_pointer_cast(d_row_offsets.data()),
            thrust::raw_pointer_cast(d_destinations.data()),
            thrust::raw_pointer_cast(d_lengths.data()),
            thrust::raw_pointer_cast(d_capacities.data()),
            num_cities,
            thrust::raw_pointer_cast(d_populated_city.data()),
            thrust::raw_pointer_cast(d_prime_age_pop.data()),
            thrust::raw_pointer_cast(d_elderly_pop.data()),
            num_populated_cities,
            thrust::raw_pointer_cast(d_shelter_capacities.data()),
            max_distance_elderly,
            path_array_dev,
            // path_len_dev,
            d_timer,
            thrust::raw_pointer_cast(d_finished.data()),
            thrust::raw_pointer_cast(current_loc.data()),
            thrust::raw_pointer_cast(next_free.data()), thrust::raw_pointer_cast(distance_travelled.data()), thrust::raw_pointer_cast(road_avilability_time.data()),
            d_road_claiming_time,
            d_road_claiming_pop,
            d_road_claiming_tid,
            thrust::raw_pointer_cast(num_drops_gpu.data()),
            d_drops
            // drops_dev
        );
        cudaDeviceSynchronize();
        // h_finished_vec = d_finished;
        // printf("fadsfsd");
        int finished_count = thrust::count(d_finished.begin(), d_finished.end(), 1);
        // printf("%d\n", finished_count);
        // End loop when all cities have finished
        if (finished_count == num_populated_cities || (difftime(time(NULL), start_time) > 480.0))
        {
            break;
        }
        // Event driven simulation:simulate only the times which have events that is some city is free to make decision
        auto out_end = thrust::copy_if(
            next_free.begin(),
            next_free.end(),
            zip_begin,
            valid_times.begin(),
            IsValidNextFree(simulation_time));
        valid_times.resize(out_end - valid_times.begin());
        if (!valid_times.empty())
        {
            simulation_time = *thrust::min_element(valid_times.begin(), valid_times.end());
        }
        else
        {
            simulation_time++;
        }

        // simulation_time += 12;
        // simulation_time++; // 1 unite of time=12 min of sim time
        // printf("%d",finished_count);
    }
    // printf("simulation time:%d\n", simulation_time);
    // timeout or unable to reach shelter drop at current location itself
    drop_rest<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(d_row_offsets.data()),
        thrust::raw_pointer_cast(d_destinations.data()),
        thrust::raw_pointer_cast(d_lengths.data()),
        thrust::raw_pointer_cast(d_capacities.data()),
        num_cities,
        thrust::raw_pointer_cast(d_populated_city.data()),
        thrust::raw_pointer_cast(d_prime_age_pop.data()),
        thrust::raw_pointer_cast(d_elderly_pop.data()),
        num_populated_cities,
        thrust::raw_pointer_cast(d_shelter_capacities.data()),
        max_distance_elderly,
        path_array_dev,
        thrust::raw_pointer_cast(path_len_dev.data()),
        d_timer,
        thrust::raw_pointer_cast(d_finished.data()),
        thrust::raw_pointer_cast(current_loc.data()),
        thrust::raw_pointer_cast(next_free.data()), thrust::raw_pointer_cast(distance_travelled.data()), thrust::raw_pointer_cast(road_avilability_time.data()),
        d_road_claiming_time,
        d_road_claiming_pop,
        d_road_claiming_tid,
        thrust::raw_pointer_cast(num_drops_gpu.data()),
        d_drops
        //  drops_dev
        // device_vectors
    );

    cudaDeviceSynchronize();
    // thrust::device_vector<int> path_len_reduce = current_loc;
    // int sum = thrust::reduce(current_loc.begin(), current_loc.end(), 0, thrust::plus<int>());

    // thrust::exclusive_scan(current_loc.begin(), current_loc.end(), path_len_reduce.begin());
    // int *d_path;
    // cudaMalloc(&d_path, sizeof(int) * sum);

    // add_path<<<blocksPerGrid, threadsPerBlock>>>(num_populated_cities,
    //                                              thrust::raw_pointer_cast(current_loc.data()), d_path, path_ptrs_dev, thrust::raw_pointer_cast(path_len_reduce.data()));

    // int *h_path = new int[sum];

    // // Copy the result from device to host
    // cudaMemcpy(h_path, d_path, sizeof(int) * sum, cudaMemcpyDeviceToHost);
    // thrust::host_vector<thrust::host_vector<int>> paths_per_city(num_populated_cities);
    // for (int city = 0; city < num_populated_cities; ++city)
    // {
    //     int start_idx = path_len_reduce[city];
    //     int end_idx = (city == num_populated_cities - 1) ? sum : path_len_reduce[city + 1];

    //     paths_per_city[city] = thrust::host_vector<int>(h_path + start_idx, h_path + end_idx);
    // }

    // // Now you can access paths_per_city[city] easily
    // for (int city = 0; city < num_populated_cities; ++city)
    // {
    //     std::cout << "Path for city " << city << ": ";
    //     for (int val : paths_per_city[city])
    //     {
    //         std::cout << val << " ";
    //     }
    //     std::cout << std::endl;
    // }

    // Allocate memory for our final output structures
    long long *path_size = new long long[num_populated_cities];
    int **paths = new int *[num_populated_cities];
    long long *num_drops = new long long[num_populated_cities];
    long long ***drops = new long long **[num_populated_cities];

    // Prepare host arrays for data transfer
    // int *path_ptrs_host = (int *)malloc(num_populated_cities * sizeof(int *));
    // Copy path pointers from device to host

    // Get drop counts and current location info from device
    thrust::host_vector<int> num_drops_host = num_drops_gpu;
    thrust::host_vector<int> current_loc_host = current_loc;
    // cudaMemcpy(path_ptrs_host, path_array_dev, num_populated_cities * sizeof(int), cudaMemcpyDeviceToHost);
    int total_size = num_populated_cities * h_max_path_length;
    int *path_array_host = (int *)malloc(total_size * sizeof(int));

    cudaMemcpy(path_array_host, path_array_dev, total_size * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the flattened path array
    for (int i = 0; i < num_populated_cities; i++)
    {

        path_size[i] = current_loc_host[i];
        int flat_index = i * h_max_path_length;
        paths[i] = &path_array_host[flat_index];
        // printf("City %d: ", i);
        // for (int j = 0; j < current_loc_host[i]; j++)
        // {
        //     printf("%d ", paths[i][j]);
        // }
        // printf("\n");
    }

    // free(path_array_host);
    // // Process path data for each city
    // for (int i = 0; i < num_populated_cities; i++)
    // {
    //     // Store the path size for this city
    //     path_size[i] = current_loc_host[i];

    //     // Allocate memory for the path data for this city
    //     paths[i] = new long long[path_size[i]];

    //     // Temporary array to hold path data from device
    //     int *path_array_host = (int *)malloc(h_max_path_length * num_cities * sizeof(int));
    //     if (path_array_host == NULL)
    //     {
    //         printf("Error: Failed to allocate memory for path_array_host[%d].\n", i);
    //         // Handle error - set empty path
    //         path_size[i] = 0;
    //         paths[i] = new long long[1]; // Minimal allocation to avoid null pointer
    //         continue;
    //     }

    //     // Calculate the offset into the flattened array for this city's path data
    //     int offset = i * h_max_path_length * num_cities;

    //     // Copy the path data from device to host using the flattened array
    //     cudaMemcpy(path_array_host, path_array_dev + offset, h_max_path_length * num_cities * sizeof(int), cudaMemcpyDeviceToHost);

    //     // Copy relevant path data to our output structure for this city
    //     for (int j = 0; j < path_size[i]; j++)
    //     {
    //         paths[i][j] = path_array_host[j];
    //     }

    //     // Free the temporary array
    //     free(path_array_host);
    // }

    // Process drop data for each city
    long long *h_drops_flat = new long long[num_populated_cities * h_max_drops * 3];
    cudaMemcpy(h_drops_flat, d_drops, num_populated_cities * h_max_drops * 3 * sizeof(long long), cudaMemcpyDeviceToHost);

    // long long ***h_drops_host = new long long **[num_populated_cities];

    for (int i = 0; i < num_populated_cities; ++i)
    {
        int drops_i = num_drops_host[i]; // Number of drops for this city
        drops[i] = new long long *[drops_i];
        num_drops[i] = num_drops_host[i];
        for (int j = 0; j < drops_i; ++j)
        {
            // Calculate base index in flat array
            int flat_index = (i * h_max_drops + j) * 3;
            drops[i][j] = &h_drops_flat[flat_index];
        }
    }
    // for (int i = 0; i < num_populated_cities; ++i)
    // {
    //     // std::cout << "City " << i << " has " << num_drops_host[i] << " drops:\n";

    //     for (int j = 0; j < num_drops_host[i]; ++j)
    //     {
    //         long long *drop = drops[i][j];
    //         // std::cout << "  Drop " << j << ": "
    //         //           << drop[0] << ", " << drop[1] << ", " << drop[2] << "\n";
    //     }
    // }

    // Now write to file as specified
    ofstream outfile(argv[2]);
    if (!outfile)
    {
        cerr << "Error: Cannot open file " << argv[2] << "\n";
        return 1;
    }

    // Write path data
    for (long long i = 0; i < num_populated_cities; i++)
    {
        long long currentPathSize = path_size[i];
        for (long long j = 0; j < currentPathSize; j++)
        {
            outfile << paths[i][j] << " ";
        }
        outfile << "\n";
    }

    // Write drop data
    for (long long i = 0; i < num_populated_cities; i++)
    {
        long long currentDropSize = num_drops[i];
        for (long long j = 0; j < currentDropSize; j++)
        {
            for (int k = 0; k < 3; k++)
            {
                outfile << drops[i][j][k] << " ";
            }
        }
        outfile << "\n";
    }

    // Close the file
    outfile.close();

    // Clean up allocated memory for our output structures
    // for (long long i = 0; i < num_populated_cities; i++)
    // {
    //     delete[] paths[i];

    //     for (long long j = 0; j < num_drops[i]; j++)
    //     {
    //         delete[] drops[i][j];
    //     }
    //     delete[] drops[i];
    // }
    delete[] path_size;
    delete[] paths;
    delete[] num_drops;
    delete[] drops;

    // // Clean up CUDA memory
    // // Free drop data on device
    // int **h_path_ptrs = new int *[num_populated_cities];
    // cudaMemcpy(h_path_ptrs, path_ptrs_dev, num_populated_cities * sizeof(int *), cudaMemcpyDeviceToHost);
    // for (int i = 0; i < num_populated_cities; ++i)
    // {
    //     cudaFree(h_path_ptrs[i]); // Free each city's path array
    // }
    // cudaFree(path_ptrs_dev); // Free the main path_ptrs_dev array
    // delete[] h_path_ptrs;

    cudaFree(d_timer);

    return 0;
}
