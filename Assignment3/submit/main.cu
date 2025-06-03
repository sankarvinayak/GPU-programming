
#include <chrono>
#include <cuda.h>
#include <fstream>
#include <iostream>
#include <math.h>

#define MOD 1000000007
using std::cin;
using std::cout;

struct Edge
{
    int src, dest, weight;
};

__device__ void atomicAddMod(unsigned *addr, int val)
{
    int old = *addr;
    int got;
    do
    {
        // printf("Spinner\n");
        got = old;
        int updated = (old + val) % MOD;
        old = atomicCAS(addr, got, updated);

    } while (old != got);
}

__device__ __forceinline__ unsigned long long combine_wt_id(unsigned wt, unsigned id) // encode the weight and id into a single ULL so that the atomic can be used
{
    return ((unsigned long long)wt << 32) | id;
}
// __device__ __forceinline__ unsigned int get_wt(unsigned long long combined)
// {
//     return combined >> 32;
// }

__device__ __forceinline__ unsigned int get_id(unsigned long long combined)
{
    return combined;
}

__global__ void init_arr(unsigned V, unsigned *comp_id)
{
    unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < V)
    {
        comp_id[id] = id;
    }
}

// __global__ void reset_min_edge(int V, unsigned long long *combined_edge_id)
// {
//     unsigned id = blockDim.x * blockIdx.x + threadIdx.x;
//     if (id < V)
//     {
//         combined_edge_id[id] = 0xFFFFFFFFFFFFFFFFULL;
//     }
// }

// union & find
__device__ int find(unsigned *compnent, int s)
{
    while (s != compnent[s]) // recursion as iterative
    {
        unsigned cmp = compnent[compnent[s]];
        compnent[s] = cmp;
        s = cmp;
    }
    return s;
}

__device__ void unite(unsigned *component, unsigned *rank, int s, int d)
{
    unsigned cmp_s = find(component, s);
    unsigned cmp_d = find(component, d);
    // unsigned old;
    while (cmp_d != cmp_s)
    {
        if (rank[cmp_s] > rank[cmp_d])
        {
            // old = atomicCAS(&component[cmp_d], cmp_d, cmp_s);
            // if (old == cmp_d)
            if (atomicCAS(&component[cmp_d], cmp_d, cmp_s) == cmp_d)
                break;
        }

        else if (rank[cmp_s] < rank[cmp_d])
        {
            // old = atomicCAS(&component[cmp_s], cmp_s, cmp_d);
            // if (old == cmp_s)
            if (atomicCAS(&component[cmp_s], cmp_s, cmp_d) == cmp_s)
                break;
        }

        else
        {
            // old = atomicCAS(&component[cmp_d], cmp_d, cmp_s);
            // if (old == cmp_d)
            if (atomicCAS(&component[cmp_d], cmp_d, cmp_s) == cmp_d)
            {
                atomicAdd(&rank[cmp_s], 1);
                // rank[cmp_s]++;
                break;
            }
        }
        cmp_s = find(component, s);
        cmp_d = find(component, d);
    }
}

__global__ void find_min_edge_of_component(unsigned V, unsigned E, Edge *edges, unsigned *comp_id, unsigned long long *combined_edge_id)
{
    unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < E)
    {
        Edge e = edges[id];
        unsigned comp_s = find(comp_id, e.src);
        unsigned comp_d = find(comp_id, e.dest);
        if (comp_s != comp_d)
        {
            unsigned long long combine_id = combine_wt_id(e.weight, id);
            if (combined_edge_id[comp_s] > combine_id)
                atomicMin(&combined_edge_id[comp_s], combine_id);
            if (combined_edge_id[comp_d] > combine_id)
                atomicMin(&combined_edge_id[comp_d], combine_id);
        }
        // unsigned wt = edges[id].weight;
        // unsigned s = edges[id].src;
        // unsigned d = edges[id].dest;
        // if (comp_id[s] != comp_id[d])
        // {
        //     if (min_edge_i[comp_id[s]] == -1 || edges[min_edge_i[comp_id[s]]].weight > wt)
        //     {
        //         atomicMin(&min_edge_i[comp_id[s]], id);
        //     }
        //     if (min_edge_i[comp_id[d]] == -1 || edges[min_edge_i[comp_id[d]]].weight > wt)
        //     {
        //         atomicMin(&min_edge_i[comp_id[d]], id);
        //     }
        // }
    }
}

__global__ void brovska(unsigned V, unsigned E, Edge *edges, unsigned *ans, unsigned *comp_id, unsigned long long *combined_edge_id, unsigned *added_list, unsigned *ncomp, unsigned int *rank_arr)
{
    unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ unsigned s_ans, s_subs;
    unsigned long long loc_combined_edge_id;
    if (threadIdx.x == 0)
    {
        s_ans = 0;
    }
    else if (threadIdx.x == 1023)
    {
        s_subs = 0;
    }
    __syncthreads();
    // if (id < V && combined_edge_id[id] != 0xFFFFFFFFFFFFFFFF)
    if (id < V)
    {
        loc_combined_edge_id = combined_edge_id[id];
        if (loc_combined_edge_id != 0xFFFFFFFFFFFFFFFF)
        {
            // unsigned wt = get_wt(loc_combined_edge_id);
            unsigned i = get_id(loc_combined_edge_id);
            // Edge ed = edges[min_edge_i[id]];

            // if (added_list[i] == 0 && atomicExch(&added_list[i], 1) == 0) // edge get addded by one endpoint
            // if (atomicExch(&added_list[i], 1) == 0)
            if (!atomicExch(&added_list[i], 1))
            {
                Edge ed = edges[i];
                atomicAddMod(&s_ans, ed.weight);
                // atomicAddMod(&s_ans, edges[i].weight);
                // atomicAdd(&s_subs, 1);
                atomicInc(&s_subs, 1025);
                unite(comp_id, rank_arr, ed.src, ed.dest);
                // unite(comp_id, rank_arr, edges[i].src, edges[i].dest);
            }
        }

        // unsigned min_comp = min(comp_id[ed.src], comp_id[ed.dest]);
        // comp_id[id] = min_comp;
        // min_edge_i[id] = -1;
    }
    __syncthreads();
    if (threadIdx.x == 0)
    {
        atomicAddMod(ans, s_ans);
    }
    else if (threadIdx.x == 1023)
    {
        atomicSub(ncomp, s_subs);
    }
}

// __global__ void sumup(unsigned E, Edge *edges, unsigned *ans, unsigned *added_list)
// {
//     __shared__ unsigned sum;
//     if (threadIdx.x == 0)
//     {
//         sum = 0;
//     }
//     __syncthreads();
//     unsigned id = threadIdx.x + blockDim.x * blockIdx.x;
//     if (id < E && added_list[id])
//     {
//         atomicAddMod(&sum, edges[id].weight);
//     }
//     __syncthreads();
//     atomicAddMod(ans, sum);
// }
__global__ void path_compress(int V, unsigned *comp_id)
{
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    if (id < V)
    {
        comp_id[id] = find(comp_id, id);
    }
}

// __global__ void brovskas(unsigned V, unsigned E, Edge *edges, unsigned *ans, unsigned *comp_id, unsigned *min_edge_i)
// {
//     int id = blockIdx.x * blockDim.x + threadIdx.x;
//     if (id < E)
//     {
//         unsigned comp_s = comp_id[edges[id].src];
//         unsigned comp_d = comp_id[edges[id].dest];
//         unsigned cost = edges[id].weight;
//         if (comp_d != comp_s)
//         {
//             if (min_edge_i[comp_s] > cost || min_edge_i[comp_s] == -1)
//             {
//                 atomicMax(&min_edge_i[comp_s], id);
//             }
//             if (min_edge_i[comp_d] > cost || min_edge_i[comp_d] == -1)
//             {
//                 atomicMax(&min_edge_i[comp_d], id);
//             }
//         }
//     }
// }
// __global__ void union_find(unsigned V, unsigned E, Edge *edges, unsigned *ans, unsigned *comp_id, unsigned *min_edge_i, unsigned *update)
// {
//     int id = blockIdx.x * blockDim.x + threadIdx.x;
//     if (id < V)
//     {
//         unsigned min_ed_id = min_edge_i[id];
//         if (min_ed_id != -1)
//         {
//             unsigned s = edges[min_ed_id].src;
//             unsigned d = edges[min_ed_id].dest;
//             unsigned comp_s = comp_id[s];
//             unsigned comp_d = comp_id[d];
//             if (comp_s != comp_d)
//             {
//                 *update = 1;
//                 unsigned max_c = max(comp_s, comp_d);
//                 comp_id[s] = max_c;
//                 comp_id[d] = max_c;
//             }
//         }
//     }
// }

int main()
{
    // Create a sample graph
    int V;
    cin >> V;
    int E;
    cin >> E;
    Edge *edges = (Edge *)malloc(sizeof(Edge) * E);
    for (int i = 0; i < E; i++)
    {
        int u, v, wt;
        std::string s;
        cin >> u >> v >> wt;
        edges[i].src = u;
        edges[i].dest = v;
        cin >> s;
        if (s == "green")
        {
            edges[i].weight = wt * 2;
        }
        else if (s == "traffic")
        {
            edges[i].weight = wt * 5;
        }
        else if (s == "dept")
        {
            edges[i].weight = wt * 3;
        }
        else
        {
            edges[i].weight = wt;
        }
    }
    Edge *d_edges;
    cudaMalloc(&d_edges, sizeof(Edge) * E);
    cudaMemcpy(d_edges, edges, sizeof(Edge) * E, cudaMemcpyHostToDevice);

    unsigned ans, *dans;
    cudaMalloc(&dans, sizeof(unsigned));
    cudaMemset(dans, 0, sizeof(unsigned));

    unsigned *d_comp_id;
    //  *d_min_edge_i;

    cudaMalloc(&d_comp_id, sizeof(unsigned) * V);
    // cudaMalloc(&d_min_edge_i, sizeof(unsigned) * V);

    unsigned *d_added_edges;
    cudaMalloc(&d_added_edges, sizeof(int) * E);
    cudaMemset(d_added_edges, 0, sizeof(unsigned) * E);

    unsigned n = V, *d_n;

    cudaMalloc(&d_n, sizeof(unsigned));

    // cudaMallocManaged(&nd, sizeof(unsigned));
    // cudaMemset(nd, V, sizeof(unsigned));

    cudaMemcpy(d_n, &n, sizeof(unsigned), cudaMemcpyHostToDevice);

    unsigned long long *combined_edge_id;
    cudaMalloc(&combined_edge_id, sizeof(unsigned long long) * V);

    unsigned *d_rank_arr;
    cudaMalloc(&d_rank_arr, sizeof(unsigned) * V);
    cudaMemset(d_rank_arr, 0, sizeof(unsigned) * V);

    unsigned blocksize = 1024;

    // cudaMemset(d_min_edge_i, -1, sizeof(int) * V);
    unsigned grid_dimE = (E + blocksize - 1) / blocksize;
    unsigned grid_dimV = (V + blocksize - 1) / blocksize;

    // Answer should be calculated in Kernel. No operations should be performed here.
    // Only copy data to device, kernel call, copy data back to host, and print the answer.

    auto start = std::chrono::high_resolution_clock::now();
    // Kernel call(s) here

    init_arr<<<grid_dimV, blocksize>>>(V, d_comp_id);
    do
    {
        // printf("%d\n", n);
        // reset_min_edge<<<grid_dimV, blocksize>>>(V, combined_edge_id);

        cudaMemset(combined_edge_id, 0xFF, sizeof(unsigned long long) * V); // Reset the min edge of each component

        // cudaMemset(combined_edge_id,0xFFFFFFFFFFFFFFFF,sizeof(unsigned long long)*V);

        find_min_edge_of_component<<<grid_dimE, blocksize>>>(V, E, d_edges, d_comp_id, combined_edge_id); // find min edge for each component

        brovska<<<grid_dimV, blocksize>>>(V, E, d_edges, dans, d_comp_id, combined_edge_id, d_added_edges, d_n, d_rank_arr); // add the min edge to the component

        path_compress<<<grid_dimV, blocksize>>>(V, d_comp_id); // union find path compression help in reducing long paths

        cudaMemcpy(&n, d_n, sizeof(int), cudaMemcpyDeviceToHost); // check how many components are remaining if it i 1 all vertices are conected hence terminate also act as a barrier before next iteration

    } while (n > 1); // max log n step because in each iteration number of components get atleast half due to bruvska algorithm

    // sumup<<<grid_dimE, blocksize>>>(E,d_edges,dans,added);

    cudaMemcpy(&ans, dans, sizeof(unsigned), cudaMemcpyDeviceToHost);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed1 = end - start;
    // Print only the total MST weight
    cout << ans << "\n";

    cout << elapsed1.count() << " s\n";
    cudaFree(combined_edge_id);
    cudaFree(d_n);
    cudaFree(d_rank_arr);
    cudaFree(d_added_edges);
    cudaFree(d_comp_id);
    cudaFree(dans);
    cudaFree(d_edges);
    free(edges);
    return 0;
}