/*
 * Copyright FalkorDB Ltd. 2026 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "src/graph/graph.h"
#include "src/util/arr.h"
#include "src/util/rmalloc.h"
#include "src/util/datablock/datablock.h"
#include "acutest.h"

#ifdef FALKORDB_FOCUSED_EDGE_TEST
#include <stdarg.h>
#include "src/configuration/config.h"
pthread_t redis_main_thread_id;

// Standalone runner: retain the normal delta flush threshold without pulling
// in Redis/module startup. Storage, tensors and all collectors are production C.
bool Config_Option_get(Config_Option_Field field, ...) {
	TEST_ASSERT(field == Config_DELTA_MAX_PENDING_CHANGES);
	va_list args;
	va_start(args, field);
	uint64_t *value = va_arg(args, uint64_t *);
	*value = DELTA_MAX_PENDING_CHANGES_DEFAULT;
	va_end(args);
	return true;
}
#endif

// The fixture deliberately bypasses Graph_DeleteEdges: its tensor still names
// a freed slot, exactly the inconsistent state observed in production. Keep
// real DataBlock storage and real scalar/vector Tensor entries in every test.
static bool synchronize
(
	const Graph *g,
	Delta_Matrix matrix,
	GrB_Index rows,
	GrB_Index cols
) {
	return Delta_Matrix_wait(matrix, true) == GrB_SUCCESS;
}

static Graph *fixture
(
	bool parallel
) {
#ifdef REDIS_MODULE_TARGET
	Alloc_Reset();
#endif
	GrB_init(GrB_NONBLOCKING);
	GxB_Global_Option_set(GxB_FORMAT, GxB_BY_ROW);
	GxB_Global_Option_set(GxB_HYPER_SWITCH, GxB_NEVER_HYPER);

	Graph *g = rm_calloc(1, sizeof(Graph));
	g->nodes = DataBlock_New(16, 16, sizeof(AttributeSet), NULL);
	g->edges = DataBlock_New(16, 16, sizeof(AttributeSet), NULL);
	g->relations = arr_new(Tensor, 1);
	arr_append(g->relations, Tensor_new(16, 16));
	g->SynchronizeMatrix = synchronize;
	pthread_rwlock_init(&g->_rwlock, NULL);
	Delta_Matrix_new(&g->adjacency_matrix, GrB_BOOL, 16, 16, true);
	g->stats.edge_count = arr_new(uint64_t, 1);
	arr_append(g->stats.edge_count, 3);

	for(int i = 0; i < 3; i++) {
		NodeID id;
		AttributeSet *attrs = DataBlock_AllocateItem(g->nodes, &id);
		*attrs = NULL;
	}
	for(int i = 0; i < 3; i++) {
		EdgeID id;
		AttributeSet *attrs = DataBlock_AllocateItem(g->edges, &id);
		// A live zero-property edge has a non-NULL slot whose value is NULL.
		// It must survive filtering; only missing slots are invalid.
		*attrs = NULL;
	}

	Tensor r = g->relations[0];
	Tensor_SetElement(r, 0, 1, 0);
	if(parallel) {
		Tensor_SetElement(r, 0, 1, 1);
		Tensor_SetElement(r, 0, 1, 2);
	} else {
		Tensor_SetElement(r, 0, 2, 1);
		Tensor_SetElement(r, 1, 2, 2);
	}
	DataBlock_DeleteItem(g->edges, 1);
	Tensor_SetElement(r, 2, 1, g->edges->itemCap + 10);
	Delta_Matrix_wait(r, true);
	return g;
}

static void destroy_fixture
(
	Graph *g
) {
	Delta_Matrix_free(&g->adjacency_matrix);
	arr_free(g->stats.edge_count);
	pthread_rwlock_destroy(&g->_rwlock);
	Tensor_free(&g->relations[0]);
	arr_free(g->relations);
	DataBlock_Free(g->edges);
	DataBlock_Free(g->nodes);
	rm_free(g);
	GrB_finalize();
}

static void assert_live
(
	Edge *edges,
	size_t count
) {
	TEST_CHECK(arr_len(edges) == count);
	TEST_MSG("expected %zu live edges, collected %u", count, arr_len(edges));
	for(size_t i = 0; i < arr_len(edges); i++) {
		TEST_ASSERT(edges[i].id == 0 || edges[i].id == 2);
		TEST_ASSERT(edges[i].attributes != NULL);
		TEST_ASSERT(*edges[i].attributes == NULL);
	}
}

static void check_endpoint_collection
(
	bool parallel
) {
	Graph *g = fixture(parallel);
	Edge *edges = arr_new(Edge, 4);
	Graph_GetEdgesConnectingNodes(g, 0, 1, 0, &edges);
	assert_live(edges, parallel ? 2 : 1);
	arr_clear(edges);
	Graph_GetEdgesConnectingNodes(g, 0, 2, GRAPH_NO_RELATION, &edges);
	assert_live(edges, 0);
	Graph_GetEdgesConnectingNodes(g, 2, 1, 0, &edges);
	assert_live(edges, 0);
	arr_free(edges);
	destroy_fixture(g);
}

static void test_scalar_endpoints(void) {
	check_endpoint_collection(false);
}

static void test_parallel_endpoints(void) {
	check_endpoint_collection(true);
}

static void check_incident_collection
(
	bool parallel
) {
	Graph *g = fixture(parallel);
	Node nodes[3];
	for(int i = 0; i < 3; i++) {
		nodes[i].id = i;
		nodes[i].attributes = DataBlock_GetItem(g->nodes, i);
	}
	Edge *outgoing = arr_new(Edge, 4);
	Edge *incoming = arr_new(Edge, 4);
	Graph_GetNodeEdges(g, &nodes[0], GRAPH_EDGE_DIR_OUTGOING, 0, &outgoing);
	assert_live(outgoing, parallel ? 2 : 1);
	arr_clear(outgoing);
	Graph_GetNodeEdges(g, &nodes[1], GRAPH_EDGE_DIR_INCOMING, GRAPH_NO_RELATION, &incoming);
	assert_live(incoming, parallel ? 2 : 1);
	arr_clear(incoming);

	Graph_CollectOutgoingEdges(&outgoing, g, nodes, 3);
	assert_live(outgoing, 2);
	arr_clear(outgoing);
	Graph_CollectIncomingEdges(&incoming, g, nodes, 3);
	assert_live(incoming, 2);
	arr_clear(incoming);
	Graph_CollectInOutEdges(&outgoing, &incoming, g, nodes, 3);
	assert_live(outgoing, 2);
	// Edges internal to the selected node set appear only in outgoing.
	assert_live(incoming, 0);
	arr_clear(outgoing);
	arr_clear(incoming);
	Graph_CollectInOutEdges(&outgoing, &incoming, g, &nodes[1], 1);
	assert_live(outgoing, parallel ? 0 : 1);
	assert_live(incoming, parallel ? 2 : 1);
	arr_free(outgoing);
	arr_free(incoming);
	destroy_fixture(g);
}

static void test_scalar_incident_edges(void) {
	check_incident_collection(false);
}

static void test_parallel_incident_edges(void) {
	check_incident_collection(true);
}

static void test_direct_missing_edge_lookup(void) {
	Graph *g = fixture(false);
	Edge e = {0};
	TEST_ASSERT(!Graph_GetEdge(g, 1, &e));
	TEST_ASSERT(e.attributes == NULL);
	TEST_ASSERT(!Graph_GetEdge(g, g->edges->itemCap + 10, &e));
	TEST_ASSERT(e.attributes == NULL);
	TEST_ASSERT(Graph_GetEdge(g, 0, &e));
	TEST_ASSERT(e.attributes != NULL);
	destroy_fixture(g);
}

static void test_repair_prevents_reused_slot_alias(void) {
	Graph *g = fixture(false);
	Edge *edges = arr_new(Edge, 4);
	// Reader finds the stale 0->2 reference and requests deferred validation.
	Graph_GetEdgesConnectingNodes(g, 0, 2, 0, &edges);
	assert_live(edges, 0);
	Graph_AcquireWriteLock(g);
	TEST_ASSERT(Graph_RepairDanglingEdges(g) == 2);
	TEST_ASSERT(Graph_RepairDanglingEdges(g) == 0);
	TEST_ASSERT(g->stats.edge_count[0] == 2);

	// Reuse the freed ID in a DIFFERENT pair. The old pair must stay absent.
	Edge replacement = {0};
	Graph_CreateEdge(g, 2, 0, 0, &replacement);
	TEST_ASSERT(replacement.id == 1);
	Graph_GetEdgesConnectingNodes(g, 0, 2, 0, &edges);
	TEST_ASSERT(arr_len(edges) == 0);
	Graph_GetEdgesConnectingNodes(g, 2, 0, 0, &edges);
	TEST_ASSERT(arr_len(edges) == 1);
	TEST_ASSERT(edges[0].id == replacement.id);
	GrB_Index nvals;
	GrB_OK(Delta_Matrix_nvals(&nvals, g->adjacency_matrix));
	TEST_ASSERT(nvals == 3);
	// This focused fixture has a custom synchronization callback, so release
	// its initialized lock directly rather than resetting production policy.
	g->_writelocked = false;
	pthread_rwlock_unlock(&g->_rwlock);
	arr_free(edges);
	destroy_fixture(g);
}

static void unlock_fixture(Graph *g) {
	g->_writelocked = false;
	pthread_rwlock_unlock(&g->_rwlock);
}

static void test_parallel_repair_keeps_propertyless_scalar(void) {
	Graph *g = fixture(true);
	DataBlock_DeleteItem(g->edges, 2);
	Graph_AcquireWriteLock(g);
	TEST_ASSERT(Graph_RepairDanglingEdges(g) == 3);
	Edge *edges = arr_new(Edge, 4);
	Graph_GetEdgesConnectingNodes(g, 0, 1, 0, &edges);
	assert_live(edges, 1);
	TEST_ASSERT(edges[0].id == 0);
	TEST_ASSERT(g->stats.edge_count[0] == 1);
	unlock_fixture(g);
	arr_free(edges);
	destroy_fixture(g);
}

static void test_repair_preserves_other_relation_union(void) {
	Graph *g = fixture(false);
	// Move live edge 2 into another relation at the stale edge's old pair.
	Edge moved = {.id=2, .src_id=1, .dest_id=2, .relationID=0};
	Tensor_RemoveElements(g->relations[0], &moved, 1, NULL);
	arr_append(g->relations, Tensor_new(16, 16));
	arr_append(g->stats.edge_count, 1);
	Tensor_SetElement(g->relations[1], 0, 2, 2);
	Graph_AcquireWriteLock(g);
	TEST_ASSERT(Graph_RepairDanglingEdges(g) == 2);
	bool connected = false;
	TEST_ASSERT(GrB_Matrix_extractElement_BOOL(&connected, DELTA_MATRIX_M(g->adjacency_matrix), 0, 2) == GrB_SUCCESS);
	TEST_ASSERT(connected);
	Edge *edges = arr_new(Edge, 4);
	Graph_GetEdgesConnectingNodes(g, 0, 2, GRAPH_NO_RELATION, &edges);
	assert_live(edges, 1);
	TEST_ASSERT(edges[0].relationID == 1);
	TEST_ASSERT(g->stats.edge_count[0] == 1);
	TEST_ASSERT(g->stats.edge_count[1] == 1);
	unlock_fixture(g);
	arr_free(edges);
	Tensor_free(&g->relations[1]);
	destroy_fixture(g);
}

TEST_LIST = {
	{"scalarDanglingEndpointEdges", test_scalar_endpoints},
	{"parallelDanglingEndpointEdges", test_parallel_endpoints},
	{"scalarDanglingIncidentEdges", test_scalar_incident_edges},
	{"parallelDanglingIncidentEdges", test_parallel_incident_edges},
	{"directMissingEdgeLookup", test_direct_missing_edge_lookup},
	{"repairPreventsReusedSlotAlias", test_repair_prevents_reused_slot_alias},
	{"parallelRepairKeepsPropertylessScalar", test_parallel_repair_keeps_propertyless_scalar},
	{"repairPreservesOtherRelationUnion", test_repair_preserves_other_relation_union},
	{NULL, NULL}
};
